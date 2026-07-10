"""Ranking ablation + cascade (retrieve → rank) evaluation.

Ablation trains the ranker with components toggled off (multimodal / DIN / DCN cross / multi-task)
and reports GAUC overall + on a cold-start slice. Cascade re-ranks the two-tower's retrieved
candidates and re-ranks them, measuring NDCG@10 / Hit@10 lift over the raw retrieval order.
"""

from __future__ import annotations

import json
import math

import numpy as np
import torch

from ..paths import Paths
from ..utils import get_logger, pick_device
from .train import evaluate_ranking, train

log = get_logger("vlmrec.ranking.eval")

ABLATIONS = [
    ("full", dict(use_multimodal=True, use_din=True, use_cross=True, use_mmoe=True, n_tasks=2)),
    (
        "no_multimodal",
        dict(use_multimodal=False, use_din=False, use_cross=True, use_mmoe=True, n_tasks=2),
    ),
    ("no_din", dict(use_multimodal=True, use_din=False, use_cross=True, use_mmoe=True, n_tasks=2)),
    (
        "no_cross",
        dict(use_multimodal=True, use_din=True, use_cross=False, use_mmoe=True, n_tasks=2),
    ),
    (
        "single_task",
        dict(use_multimodal=True, use_din=True, use_cross=True, use_mmoe=False, n_tasks=1),
    ),
]


def _cold_users(d):
    test_users = np.where(d.test_item >= 0)[0]
    tp = d.item_pop[d.test_item[test_users]]
    thr = int(np.quantile(tp, 0.25))
    return test_users[tp <= thr], thr


def _ndcg10(rank: int) -> float:
    return 1.0 / math.log2(rank + 2) if rank < 10 else 0.0


def _minmax(x: np.ndarray) -> np.ndarray:
    lo, hi = float(x.min()), float(x.max())
    return (x - lo) / (hi - lo + 1e-9)


@torch.no_grad()
def _cascade_eval(
    cfg, paths, full_bundle, device, k=200, sample=3000, seed=0, target="test", alphas=None
) -> dict:
    """Re-rank the two-tower's top-K with the ranker; compare NDCG@10/Hit@10 by order.

    ``target`` picks the held-out split ("test" or "valid" — valid is for tuning fusion alpha
    without touching test). ``alphas`` adds fused orders alpha*retrieval + (1-alpha)*ranker
    (both min-max normalized per user); alpha=1 degenerates to retrieval order.
    """
    from ..retrieval.model import TwoTower

    model, rd, (content, cat, seq_t, _) = full_bundle
    use_score = bool(getattr(model, "use_retrieval_score", False))
    d = rd.base
    targets = d.test_item if target == "test" else d.valid_item
    rdir = paths.data / "retrieval"
    item_e = torch.tensor(np.load(rdir / "item_emb_content.npy"), device=device)
    rr = cfg.retrieval
    rt = TwoTower(
        d.content_dim,
        d.n_users,
        d.cat_cardinalities,
        out_dim=int(rr.out_dim),
        hidden=tuple(rr.hidden),
        feature_mode="content",
        temperature=float(rr.temperature),
    ).to(device)
    rt.load_state_dict(torch.load(rdir / "model_content.pt", map_location=device))
    rt.eval()
    usum = torch.tensor(d.user_sum_content, device=device)
    ucnt = torch.tensor(d.user_count, device=device)
    rcontent = content[:-1]  # drop the pad row -> (N, Cd), matches retrieval

    target_users = np.where(targets >= 0)[0]
    rng = np.random.default_rng(seed)
    users = rng.choice(target_users, size=min(sample, len(target_users)), replace=False)

    alphas = list(alphas or [])
    ndcg_r, ndcg_k, hit_r, hit_k, found = [], [], [], [], 0
    ndcg_a: dict[float, list] = {a: [] for a in alphas}
    for s0 in range(0, len(users), 512):
        ub = users[s0 : s0 + 512]
        b = len(ub)
        ue = rt.user_embeddings_eval(torch.as_tensor(ub, device=device), usum, ucnt, rcontent)
        scores = ue @ item_e.t()
        rows, cols = [], []
        for r, u in enumerate(ub):
            a, bb = d.seen_indptr[u], d.seen_indptr[u + 1]
            its = d.seen_items[a:bb]
            rows.append(np.full(len(its), r))
            cols.append(its)
        flat = torch.as_tensor(
            np.concatenate(rows) * d.n_items + np.concatenate(cols), device=device
        )
        scores.view(-1)[flat] = -1e9
        top = torch.topk(scores, k, dim=1)
        topk, topv = top.indices, top.values  # (b, k), retrieval-descending

        seq_b = seq_t[torch.as_tensor(np.repeat(ub, k), device=device)]
        ret = topv.reshape(-1).float() if use_score else None
        rk = model(topk.reshape(-1), seq_b, content, cat, ret_score=ret)[:, 0].reshape(b, k)
        rk_order = torch.argsort(rk, dim=1, descending=True).cpu().numpy()
        rk_np = torch.sigmoid(rk).float().cpu().numpy()
        topk_np = topk.cpu().numpy()
        topv_np = topv.float().cpu().numpy()
        for r, u in enumerate(ub):
            pos = np.where(topk_np[r] == int(targets[u]))[0]
            if len(pos) == 0:
                continue
            found += 1
            ret_rank = int(pos[0])
            rank_pos = int(np.where(rk_order[r] == ret_rank)[0][0])
            hit_r.append(1 if ret_rank < 10 else 0)
            hit_k.append(1 if rank_pos < 10 else 0)
            ndcg_r.append(_ndcg10(ret_rank))
            ndcg_k.append(_ndcg10(rank_pos))
            if alphas:
                rn, kn = _minmax(topv_np[r]), _minmax(rk_np[r])
                for a in alphas:
                    fused_order = np.argsort(-(a * rn + (1 - a) * kn), kind="stable")
                    ndcg_a[a].append(_ndcg10(int(np.where(fused_order == ret_rank)[0][0])))

    nf = max(found, 1)
    out = {
        "K": k,
        "sample": int(len(users)),
        "target": target,
        "recall@K": round(found / len(users), 5),
        "ndcg@10_retrieval_order": round(float(np.sum(ndcg_r) / nf), 5),
        "ndcg@10_ranker_order": round(float(np.sum(ndcg_k) / nf), 5),
        "hit@10_retrieval_order": round(float(np.sum(hit_r) / nf), 5),
        "hit@10_ranker_order": round(float(np.sum(hit_k) / nf), 5),
    }
    if alphas:
        out["ndcg@10_fused"] = {str(a): round(float(np.sum(v) / nf), 5) for a, v in ndcg_a.items()}
    return out


def _write_md(out_dir, s) -> None:
    lines = [
        "# Week 3 — Ranking ablation",
        "",
        "Test split; **GAUC** = per-user AUC. Cold-start = test users whose target "
        f"train-pop ≤ {s['cold_threshold_pop']}.",
        "",
        "| variant | click GAUC | click AUC | LogLoss | sat AUC | GAUC (cold) |",
        "|---|---|---|---|---|---|",
    ]
    for name, m in s["results"].items():
        f = m["full"]
        lines.append(
            f"| {name} | {f['click_GAUC']:.4f} | {f['click_AUC']:.4f} | {f['click_LogLoss']:.4f} | "
            f"{f.get('sat_AUC', float('nan')):.4f} | {m['cold_GAUC']:.4f} |"
        )
    if s.get("cascade"):
        c = s["cascade"]
        lines += [
            "",
            "## Cascade: retrieve (two-tower) → re-rank",
            f"On a {c['sample']}-user sample, recall@{c['K']} = {c['recall@K']:.3f}.",
            f"For retrieved positives, ranking lifts NDCG@10 "
            f"{c['ndcg@10_retrieval_order']:.3f} -> {c['ndcg@10_ranker_order']:.3f}, "
            f"Hit@10 {c['hit@10_retrieval_order']:.3f} -> {c['hit@10_ranker_order']:.3f}.",
        ]
    (out_dir / "WEEK3_RESULTS.md").write_text("\n".join(lines) + "\n")


def run(cfg, paths: Paths) -> dict:
    r = cfg.ranking
    device = pick_device(str(cfg.device))
    out_dir = paths.data / "ranking"
    out_dir.mkdir(parents=True, exist_ok=True)
    from ..retrieval.data import cfg_sources

    common = dict(
        epochs=int(r.epochs),
        batch_size=int(r.batch_size),
        lr=float(r.lr),
        n_neg=int(r.n_neg_train),
        n_neg_eval=int(r.n_neg_eval),
        d_model=int(r.d_model),
        max_seq_len=int(r.max_seq_len),
        seed=int(cfg.seed),
        sources=cfg_sources(cfg),
    )

    results, full_bundle, thr = {}, None, None
    for name, flags in ABLATIONS:
        model, rd, tensors, m = train(cfg, paths, label=name, **flags, **common)
        d = rd.base
        cold, thr = _cold_users(d)
        content, cat, seq_t, ret_ui = tensors
        m_cold = evaluate_ranking(
            model,
            rd,
            content,
            cat,
            seq_t,
            device,
            n_neg=int(r.n_neg_eval),
            user_subset=cold,
            ret_ui=ret_ui,
        )
        results[name] = {"full": m, "cold_GAUC": m_cold["click_GAUC"]}
        log.info("rank[%s] GAUC full=%.4f cold=%.4f", name, m["click_GAUC"], m_cold["click_GAUC"])
        if name == "full":
            full_bundle = (model, rd, tensors)

    cascade = None
    try:
        cascade = _cascade_eval(cfg, paths, full_bundle, device)
        log.info("cascade: %s", cascade)
    except Exception as e:  # noqa: BLE001 - best-effort; ablation stands without it
        log.warning("cascade eval skipped: %s", e)

    summary = {"cold_threshold_pop": thr, "results": results, "cascade": cascade}
    (out_dir / "ablation.json").write_text(json.dumps(summary, indent=2, default=float))
    _write_md(out_dir, summary)
    log.info("ranking ablation -> %s", out_dir / "ablation.json")
    return summary
