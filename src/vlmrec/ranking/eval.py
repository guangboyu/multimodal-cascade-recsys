"""Week-3 ranking ablation + cascade (retrieve → rank) evaluation.

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


@torch.no_grad()
def _cascade_eval(cfg, paths, full_bundle, device, k=200, sample=3000, seed=0) -> dict:
    from ..retrieval.model import TwoTower

    model, rd, (content, cat, seq_t) = full_bundle
    d = rd.base
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

    test_users = np.where(d.test_item >= 0)[0]
    rng = np.random.default_rng(seed)
    users = rng.choice(test_users, size=min(sample, len(test_users)), replace=False)

    ndcg_r, ndcg_k, hit_r, hit_k, found = [], [], [], [], 0
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
        topk = torch.topk(scores, k, dim=1).indices  # (b, k)

        seq_b = seq_t[torch.as_tensor(np.repeat(ub, k), device=device)]
        rk = model(topk.reshape(-1), seq_b, content, cat)[:, 0].reshape(b, k)
        rk_order = torch.argsort(rk, dim=1, descending=True).cpu().numpy()
        topk_np = topk.cpu().numpy()
        for r, u in enumerate(ub):
            pos = np.where(topk_np[r] == int(d.test_item[u]))[0]
            if len(pos) == 0:
                continue
            found += 1
            ret_rank = int(pos[0])
            rank_pos = int(np.where(rk_order[r] == ret_rank)[0][0])
            hit_r.append(1 if ret_rank < 10 else 0)
            hit_k.append(1 if rank_pos < 10 else 0)
            ndcg_r.append(1.0 / math.log2(ret_rank + 2) if ret_rank < 10 else 0.0)
            ndcg_k.append(1.0 / math.log2(rank_pos + 2) if rank_pos < 10 else 0.0)

    nf = max(found, 1)
    return {
        "K": k,
        "sample": int(len(users)),
        "recall@K": round(found / len(users), 5),
        "ndcg@10_retrieval_order": round(float(np.sum(ndcg_r) / nf), 5),
        "ndcg@10_ranker_order": round(float(np.sum(ndcg_k) / nf), 5),
        "hit@10_retrieval_order": round(float(np.sum(hit_r) / nf), 5),
        "hit@10_ranker_order": round(float(np.sum(hit_k) / nf), 5),
    }


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
    common = dict(
        epochs=int(r.epochs),
        batch_size=int(r.batch_size),
        lr=float(r.lr),
        n_neg=int(r.n_neg_train),
        n_neg_eval=int(r.n_neg_eval),
        d_model=int(r.d_model),
        max_seq_len=int(r.max_seq_len),
        seed=int(cfg.seed),
    )

    results, full_bundle, thr = {}, None, None
    for name, flags in ABLATIONS:
        model, rd, tensors, m = train(cfg, paths, label=name, **flags, **common)
        d = rd.base
        cold, thr = _cold_users(d)
        m_cold = evaluate_ranking(
            model, rd, *tensors, device, n_neg=int(r.n_neg_eval), user_subset=cold
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
