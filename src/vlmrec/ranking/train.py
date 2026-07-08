"""Train the ranker (multi-task BCE with sampled negatives); eval with GAUC / AUC / LogLoss.

Each training positive is paired with N random negatives; the model predicts P(click) and
P(satisfaction). Eval scores each test user's held-out positive against `n_neg` sampled negatives:
  * **GAUC** — per-user AUC averaged over users (the metric that matters in ranking)
  * **AUC**  — pooled positives-vs-negatives
  * **LogLoss** on the click head
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..paths import Paths
from ..utils import get_logger, pick_device, set_seed
from .data import build_ranking_data, sample_negatives
from .model import Ranker

log = get_logger("vlmrec.ranking.train")


def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    n_p, n_n = len(pos), len(neg)
    if n_p == 0 or n_n == 0:
        return float("nan")
    all_s = np.concatenate([pos, neg])
    # Mann-Whitney with midranks: tied scores share the average of their ordinal ranks,
    # so a tie counts 0.5 instead of an argsort-order coin flip.
    _, inv, counts = np.unique(all_s, return_inverse=True, return_counts=True)
    order = all_s.argsort()
    ordinal = np.empty(len(all_s))
    ordinal[order] = np.arange(1, len(all_s) + 1)
    ranks = (np.bincount(inv, weights=ordinal) / counts)[inv]
    return float((ranks[:n_p].sum() - n_p * (n_p + 1) / 2) / (n_p * n_n))


@torch.no_grad()
def evaluate_ranking(
    model,
    rd,
    content_pad,
    cat,
    seq_t,
    device,
    n_neg=100,
    user_subset=None,
    batch_users=512,
    seed=0,
    ret_ui=None,
) -> dict:
    d = rd.base
    targets = d.test_item
    if user_subset is None:
        users = np.where(targets >= 0)[0]
    else:
        users = np.asarray(user_subset, dtype=np.int64)
        users = users[targets[users] >= 0]
    rng = np.random.default_rng(seed)
    model.eval()
    pos_c, neg_c, gauc, pos_s, neg_s = [], [], [], [], []
    for s in range(0, len(users), batch_users):
        ub = users[s : s + batch_users]
        b = len(ub)
        negs = rng.integers(0, d.n_items, size=(b, n_neg))
        cands = np.concatenate([targets[ub][:, None], negs], axis=1)  # (b, 1+n_neg)
        cand_t = torch.as_tensor(cands.reshape(-1), device=device)
        users_t = torch.as_tensor(np.repeat(ub, 1 + n_neg), device=device)
        seq_b = seq_t[users_t]
        ret = None
        if ret_ui is not None:
            ue, ie = ret_ui
            ret = (ue[users_t] * ie[cand_t]).sum(-1)  # exact retrieval dot product per pair
        logits = model(cand_t, seq_b, content_pad, cat, ret_score=ret)
        click = torch.sigmoid(logits[:, 0]).float().cpu().numpy().reshape(b, 1 + n_neg)
        ps, ns = click[:, 0], click[:, 1:]
        pos_c.append(ps)
        neg_c.append(ns.reshape(-1))
        gauc.extend(((ps[:, None] > ns) + 0.5 * (ps[:, None] == ns)).mean(axis=1))
        if logits.shape[1] > 1:
            sat = torch.sigmoid(logits[:, 1]).float().cpu().numpy().reshape(b, 1 + n_neg)
            pos_s.append(sat[:, 0])
            neg_s.append(sat[:, 1:].reshape(-1))

    pos_c, neg_c = np.concatenate(pos_c), np.concatenate(neg_c)
    eps = 1e-7
    logloss = -(np.log(pos_c + eps).sum() + np.log(1 - neg_c + eps).sum()) / (
        len(pos_c) + len(neg_c)
    )
    out = {
        "click_GAUC": round(float(np.mean(gauc)), 5),
        "click_AUC": round(_auc(pos_c, neg_c), 5),
        "click_LogLoss": round(float(logloss), 5),
        "n_users_eval": int(len(users)),
    }
    if pos_s:
        out["sat_AUC"] = round(_auc(np.concatenate(pos_s), np.concatenate(neg_s)), 5)
    return out


def train(
    cfg,
    paths: Paths,
    *,
    use_multimodal=True,
    use_din=True,
    use_cross=True,
    use_mmoe=True,
    epochs=3,
    batch_size=2048,
    lr=1e-3,
    n_neg=4,
    n_neg_eval=100,
    d_model=128,
    n_tasks=2,
    max_seq_len=30,
    seed=42,
    label="full",
    cand_topk=None,
    hard_neg_frac=0.0,
    use_retrieval_score=False,
    sources=("text", "image"),
    loss_type="bce",  # bce (pointwise) | softmax (listwise over the [pos + negs] slate)
):
    set_seed(seed)
    device = pick_device(str(cfg.device))
    rd = build_ranking_data(paths, max_seq_len=max_seq_len, sources=sources)
    d = rd.base
    # content padded with a zero row at index n_items (the sequence pad index)
    content = torch.tensor(
        np.vstack([d.content, np.zeros((1, d.content.shape[1]), np.float32)]), device=device
    )
    cat = torch.tensor(d.cat_features, device=device)
    seq_t = torch.tensor(rd.seq, device=device)
    pos = rd.pos
    n_pos, n_items = len(pos), d.n_items

    ret_ui = None
    if use_retrieval_score:
        # exact two-tower embeddings saved by rerank.candidates — the cross-stage score feature
        ret_ui = (
            torch.tensor(
                np.load(paths.data / "rerank" / "user_emb.npy").astype(np.float32), device=device
            ),
            torch.tensor(
                np.load(paths.data / "retrieval" / "item_emb_content.npy").astype(np.float32),
                device=device,
            ),
        )

    model = Ranker(
        d.content_dim,
        d.cat_cardinalities,
        d_model=d_model,
        n_tasks=n_tasks,
        use_multimodal=use_multimodal,
        use_din=use_din,
        use_cross=use_cross,
        use_mmoe=use_mmoe,
        use_retrieval_score=use_retrieval_score,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = np.random.default_rng(seed)
    log.info(
        "rank[%s] params=%s | mm=%s din=%s cross=%s mmoe=%s tasks=%s",
        label,
        f"{sum(p.numel() for p in model.parameters()):,}",
        use_multimodal,
        use_din,
        use_cross,
        use_mmoe,
        n_tasks,
    )

    metrics = {}
    for ep in range(1, epochs + 1):
        model.train()
        perm = rng.permutation(n_pos)
        total, nb = 0.0, 0
        t0 = time.perf_counter()
        for s in range(0, n_pos, batch_size):
            idx = perm[s : s + batch_size]
            b = len(idx)
            pi, pu, psat = pos[idx, 1], pos[idx, 0], pos[idx, 2]
            negs = sample_negatives(
                rng,
                pu,
                pi,
                n_items,
                n_neg,
                d.valid_item,
                d.test_item,
                cand_topk=cand_topk,
                hard_frac=hard_neg_frac,
            )
            cand = np.concatenate([pi[:, None], negs], axis=1).reshape(-1)
            users = np.repeat(pu, 1 + n_neg)
            click = np.zeros((b, 1 + n_neg), np.float32)
            click[:, 0] = 1.0
            cand_t = torch.as_tensor(cand, device=device)
            users_t = torch.as_tensor(users, device=device)
            seq_b = seq_t[users_t]
            ret = (ret_ui[0][users_t] * ret_ui[1][cand_t]).sum(-1) if ret_ui else None
            logits = model(cand_t, seq_b, content, cat, ret_score=ret)
            if loss_type == "softmax":
                # listwise: the positive must out-score its own slate of sampled negatives —
                # matches the serving task (order retrieved candidates), unlike pointwise BCE
                clog = logits[:, 0].reshape(b, 1 + n_neg)
                loss = F.cross_entropy(clog, torch.zeros(b, dtype=torch.long, device=device))
            else:
                loss = F.binary_cross_entropy_with_logits(
                    logits[:, 0], torch.as_tensor(click.reshape(-1), device=device)
                )
            if n_tasks > 1:
                sat = np.zeros((b, 1 + n_neg), np.float32)
                sat[:, 0] = psat
                loss = loss + F.binary_cross_entropy_with_logits(
                    logits[:, 1], torch.as_tensor(sat.reshape(-1), device=device)
                )
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        metrics = evaluate_ranking(
            model, rd, content, cat, seq_t, device, n_neg=n_neg_eval, ret_ui=ret_ui
        )
        log.info(
            "rank[%s] ep%02d | loss %.4f | %.1fs | %s",
            label,
            ep,
            total / nb,
            time.perf_counter() - t0,
            metrics,
        )

    return model, rd, (content, cat, seq_t, ret_ui), metrics


def run(cfg, paths: Paths) -> dict:
    from ..retrieval.data import cfg_sources

    r = cfg.ranking
    out_dir = paths.data / "ranking"
    out_dir.mkdir(parents=True, exist_ok=True)
    model, _, _, metrics = train(
        cfg,
        paths,
        use_multimodal=bool(r.use_multimodal),
        use_din=bool(r.use_din),
        use_cross=bool(r.use_cross),
        use_mmoe=bool(r.use_mmoe),
        epochs=int(r.epochs),
        batch_size=int(r.batch_size),
        lr=float(r.lr),
        n_neg=int(r.n_neg_train),
        n_neg_eval=int(r.n_neg_eval),
        d_model=int(r.d_model),
        n_tasks=int(r.n_tasks),
        max_seq_len=int(r.max_seq_len),
        seed=int(cfg.seed),
        label="full",
        sources=cfg_sources(cfg),
    )
    torch.save(model.state_dict(), out_dir / "model_full.pt")
    (out_dir / "metrics_full.json").write_text(json.dumps(metrics, indent=2))
    log.info("ranking full -> %s | TEST %s", out_dir / "model_full.pt", metrics)
    return metrics
