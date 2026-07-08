"""Train the two-tower retriever with in-batch sampled softmax; eval with Recall@K / NDCG@K.

Saves per-mode artifacts under ``data/retrieval/``:
  item_emb_<mode>.npy  — all item-tower embeddings (for the FAISS index / eval)
  model_<mode>.pt      — model state dict (for serving)
  metrics_<mode>.json  — valid curve + final test metrics
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from ..paths import Paths
from ..utils import get_logger, pick_device, set_seed
from .data import RetrievalData, load_retrieval_data
from .model import TwoTower

log = get_logger("vlmrec.retrieval.train")


def _tensors(d: RetrievalData, device: str) -> dict:
    return {
        "content": torch.tensor(d.content, device=device),
        "cat": torch.tensor(d.cat_features, device=device),
        "user_sum": torch.tensor(d.user_sum_content, device=device),
        "user_count": torch.tensor(d.user_count, device=device),
    }


@torch.no_grad()
def evaluate(
    model,
    d: RetrievalData,
    T: dict,
    split: str,
    ks=(10, 50, 100, 500),
    user_batch=4096,
    user_subset=None,
):
    device = T["content"].device
    item_e = model.all_item_embeddings(T["content"], T["cat"])  # (N, dim)
    n_items = item_e.shape[0]
    targets = d.valid_item if split == "valid" else d.test_item
    if user_subset is None:
        users = np.where(targets >= 0)[0]
    else:
        users = np.asarray(user_subset, dtype=np.int64)
        users = users[targets[users] >= 0]
    tgt = targets[users]
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}

    for s in range(0, len(users), user_batch):
        ub = users[s : s + user_batch]
        u_t = torch.as_tensor(ub, device=device)
        ue = model.user_embeddings_eval(u_t, T["user_sum"], T["user_count"], T["content"])
        scores = ue @ item_e.t()  # (b, N)

        # mask already-seen items (train history; +valid item when scoring test) in one scatter
        rows, cols = [], []
        for r, u in enumerate(ub):
            a, b = d.seen_indptr[u], d.seen_indptr[u + 1]
            its = d.seen_items[a:b]
            rows.append(np.full(len(its), r))
            cols.append(its)
            if split == "test" and d.valid_item[u] >= 0:
                rows.append(np.array([r]))
                cols.append(np.array([d.valid_item[u]]))
        flat = torch.as_tensor(np.concatenate(rows) * n_items + np.concatenate(cols), device=device)
        scores.view(-1)[flat] = -1e9

        topk = torch.topk(scores, max_k, dim=1).indices.cpu().numpy()
        tb = tgt[s : s + len(ub)]
        for r in range(len(ub)):
            hit = np.where(topk[r] == tb[r])[0]
            if len(hit):
                rank = int(hit[0])  # 0-indexed
                for k in ks:
                    if rank < k:
                        hits[k] += 1
                        ndcg[k] += 1.0 / np.log2(rank + 2)

    n = len(users)
    out = {f"Recall@{k}": round(hits[k] / n, 5) for k in ks}
    out |= {f"NDCG@{k}": round(ndcg[k] / n, 5) for k in ks}
    out["n_users_eval"] = int(n)
    return out


def train(
    cfg,
    paths: Paths,
    feature_mode: str = "content",
    epochs: int = 10,
    batch_size: int = 4096,
    lr: float = 2e-3,
    out_dim: int = 128,
    hidden=(256,),
    temperature: float = 0.05,
    eval_every: int = 1,
    ks=(10, 50, 100, 500),
    seed: int = 42,
    sources=("text", "image"),
    d: RetrievalData | None = None,
    save: bool = True,
    return_model: bool = False,
    sid_codes=None,  # (N, levels) — required for the sid / content_sid feature modes
):
    set_seed(seed)
    device = pick_device(str(cfg.device))
    out_dir = paths.data / "retrieval"
    out_dir.mkdir(parents=True, exist_ok=True)

    if d is None:
        d = load_retrieval_data(paths, sources=sources)
    T = _tensors(d, device)
    log_q = torch.log(
        torch.tensor(d.item_pop, dtype=torch.float32, device=device).clamp(min=1)
        / float(d.item_pop.sum())
    )
    pairs = torch.as_tensor(d.train_pairs, device=device)
    n_pairs = pairs.shape[0]

    model = TwoTower(
        content_dim=d.content_dim,
        n_users=d.n_users,
        cat_cardinalities=d.cat_cardinalities,
        out_dim=out_dim,
        hidden=tuple(hidden),
        feature_mode=feature_mode,
        temperature=temperature,
        sid_codes=None if sid_codes is None else torch.as_tensor(sid_codes, device=device),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    log.info(
        "train two-tower [mode=%s] params=%s on %s",
        feature_mode,
        f"{sum(p.numel() for p in model.parameters()):,}",
        device,
    )

    track_k = ks[min(1, len(ks) - 1)]  # track Recall@50 by default
    best = -1.0
    valid_curve = []
    for ep in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n_pairs, device=device)
        total, nb = 0.0, 0
        t0 = time.perf_counter()
        for s in range(0, n_pairs, batch_size):
            idx = perm[s : s + batch_size]
            u, i = pairs[idx, 0], pairs[idx, 1]
            ue, ie = model.train_embeddings(
                u, i, T["user_sum"], T["user_count"], T["content"], T["cat"]
            )
            loss = model.loss(ue, ie, i, log_q)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item()
            nb += 1
        msg = f"epoch {ep:02d} | loss {total / nb:.4f} | {time.perf_counter() - t0:.1f}s"
        if ep % eval_every == 0 or ep == epochs:
            model.eval()
            mv = evaluate(model, d, T, "valid", ks)
            valid_curve.append({"epoch": ep, **mv})
            log.info(
                "%s | valid R@%d=%.4f R@100=%.4f NDCG@10=%.4f",
                msg,
                track_k,
                mv[f"Recall@{track_k}"],
                mv.get("Recall@100", float("nan")),
                mv["NDCG@10"],
            )
            if mv[f"Recall@{track_k}"] > best:
                best = mv[f"Recall@{track_k}"]
                if save:
                    item_e = model.all_item_embeddings(T["content"], T["cat"]).cpu().numpy()
                    np.save(out_dir / f"item_emb_{feature_mode}.npy", item_e)
                    torch.save(model.state_dict(), out_dir / f"model_{feature_mode}.pt")
        else:
            log.info(msg)

    model.eval()
    test_m = evaluate(model, d, T, "test", ks)
    metrics = {
        "feature_mode": feature_mode,
        "sources": list(sources),
        "epochs": epochs,
        "n_params": int(sum(p.numel() for p in model.parameters())),
        "best_valid": {f"Recall@{track_k}": best},
        "valid_curve": valid_curve,
        "test": test_m,
    }
    if save:
        (out_dir / f"metrics_{feature_mode}.json").write_text(json.dumps(metrics, indent=2))
        log.info("saved -> %s", out_dir / f"item_emb_{feature_mode}.npy")
    log.info("[mode=%s] TEST %s", feature_mode, test_m)
    if return_model:
        return metrics, model, d, T
    return metrics


def run(cfg, paths: Paths) -> dict:
    from .data import cfg_sources

    r = cfg.retrieval
    return train(
        cfg,
        paths,
        feature_mode=str(r.feature_mode),
        epochs=int(r.epochs),
        batch_size=int(r.batch_size),
        lr=float(r.lr),
        out_dim=int(r.out_dim),
        hidden=tuple(r.hidden),
        temperature=float(r.temperature),
        eval_every=int(r.eval_every),
        ks=tuple(int(k) for k in r.ks),
        seed=int(cfg.seed),
        sources=cfg_sources(cfg),
    )
