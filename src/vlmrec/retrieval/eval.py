"""Week-2 retrieval evaluation + ablation.

Compares candidate-generation sources on the temporal test split — overall and on a COLD-START
(long-tail) slice (test users whose target item has low train popularity):
  * two-tower content / hybrid / id   (reloaded from checkpoints)
  * item2item co-visitation
  * popularity baseline
plus a FAISS HNSW ANN-vs-exact recall check on the content tower. Writes
``data/retrieval/ablation.json`` + ``WEEK2_RESULTS.md``.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from ..paths import Paths
from ..utils import get_logger, pick_device
from .data import load_retrieval_data
from .i2i import build_covisitation, evaluate_i2i
from .model import TwoTower
from .train import _tensors, evaluate

log = get_logger("vlmrec.retrieval.eval")


def _load_model(cfg, d, mode: str, ckpt, device):
    r = cfg.retrieval
    m = TwoTower(
        content_dim=d.content_dim,
        n_users=d.n_users,
        cat_cardinalities=d.cat_cardinalities,
        out_dim=int(r.out_dim),
        hidden=tuple(r.hidden),
        feature_mode=mode,
        temperature=float(r.temperature),
    ).to(device)
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()
    return m


def _popularity_eval(d, split="test", ks=(10, 50, 100, 500), user_subset=None) -> dict:
    """Exact: a user's effective rank for the target = global pop-rank minus seen items above it."""
    targets = d.test_item if split == "test" else d.valid_item
    if user_subset is None:
        users = np.where(targets >= 0)[0]
    else:
        users = np.asarray(user_subset, dtype=np.int64)
        users = users[targets[users] >= 0]
    order = np.argsort(-d.item_pop, kind="stable")
    rank_of = np.empty(d.n_items, dtype=np.int64)
    rank_of[order] = np.arange(d.n_items)
    indptr, items = d.seen_indptr, d.seen_items
    hits = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}
    for u in users:
        tr = int(rank_of[int(targets[u])])
        seen = items[indptr[u] : indptr[u + 1]]
        eff = tr - int((rank_of[seen] < tr).sum())  # rank among unseen, 0-indexed
        for k in ks:
            if eff < k:
                hits[k] += 1
                ndcg[k] += 1.0 / np.log2(eff + 2)
    n = len(users)
    out = {f"Recall@{k}": round(hits[k] / n, 5) for k in ks}
    out |= {f"NDCG@{k}": round(ndcg[k] / n, 5) for k in ks}
    out["n_users_eval"] = int(n)
    return out


def _write_markdown(out_dir, s) -> None:
    ks = s["ks"]
    lines = [
        "# Week 2 — Retrieval results",
        "",
        f"Temporal leave-last-out **test** split. Cold-start slice = test users whose target item "
        f"has train-pop ≤ {s['cold_start_threshold_pop']} "
        f"({s['n_cold_users']:,}/{s['n_test_users']:,} users).",
        "",
        "| source | " + " | ".join(f"R@{k}" for k in ks) + " | R@100 (cold) | NDCG@10 |",
        "|" + "---|" * (len(ks) + 3),
    ]
    for src, m in s["results"].items():
        full, cold = m["full"], m["cold"]
        row = (
            f"| {src} | "
            + " | ".join(f"{full[f'Recall@{k}']:.4f}" for k in ks)
            + f" | {cold['Recall@100']:.4f} | {full['NDCG@10']:.4f} |"
        )
        lines.append(row)
    if s.get("ann_hnsw_recall@100") is not None:
        lines += [
            "",
            f"FAISS HNSW ANN recall@100 vs exact (content): **{s['ann_hnsw_recall@100']:.4f}**",
        ]
    (out_dir / "WEEK2_RESULTS.md").write_text("\n".join(lines) + "\n")


def run(cfg, paths: Paths) -> dict:
    from .data import cfg_sources

    device = pick_device(str(cfg.device))
    d = load_retrieval_data(paths, sources=cfg_sources(cfg))
    T = _tensors(d, device)
    ks = tuple(int(k) for k in cfg.retrieval.ks)
    out_dir = paths.data / "retrieval"

    test_users = np.where(d.test_item >= 0)[0]
    target_pop = d.item_pop[d.test_item[test_users]]
    thr = int(np.quantile(target_pop, 0.25))
    cold_users = test_users[target_pop <= thr]
    log.info(
        "cold-start: target train-pop <= %d -> %d/%d test users",
        thr,
        len(cold_users),
        len(test_users),
    )

    results: dict = {}
    for mode in ["content", "hybrid", "id"]:
        ckpt = out_dir / f"model_{mode}.pt"
        if not ckpt.exists():
            continue
        m = _load_model(cfg, d, mode, ckpt, device)
        full = evaluate(m, d, T, "test", ks)
        cold = evaluate(m, d, T, "test", ks, user_subset=cold_users)
        results[f"twotower-{mode}"] = {"full": full, "cold": cold}
        log.info(
            "twotower-%-7s full R@100=%.4f | cold R@100=%.4f",
            mode,
            full["Recall@100"],
            cold["Recall@100"],
        )

    log.info("building item2item co-visitation ...")
    neigh = build_covisitation(d)
    results["i2i"] = {
        "full": evaluate_i2i(d, neigh, "test", ks),
        "cold": evaluate_i2i(d, neigh, "test", ks, user_subset=cold_users),
    }
    results["popularity"] = {
        "full": _popularity_eval(d, "test", ks),
        "cold": _popularity_eval(d, "test", ks, user_subset=cold_users),
    }

    ann = None
    if (out_dir / "item_emb_content.npy").exists() and (out_dir / "model_content.pt").exists():
        from .index import ann_vs_exact_recall

        item_e = np.load(out_dir / "item_emb_content.npy")
        mc = _load_model(cfg, d, "content", out_dir / "model_content.pt", device)
        sample = test_users[:2000]
        ue = (
            mc.user_embeddings_eval(
                torch.as_tensor(sample, device=device), T["user_sum"], T["user_count"], T["content"]
            )
            .cpu()
            .numpy()
        )
        ann = ann_vs_exact_recall(item_e, ue, k=100, kind="hnsw")
        log.info("FAISS HNSW ANN recall@100 vs exact (content): %.4f", ann)

    summary = {
        "cold_start_threshold_pop": thr,
        "n_cold_users": int(len(cold_users)),
        "n_test_users": int(len(test_users)),
        "ann_hnsw_recall@100": ann,
        "ks": list(ks),
        "results": results,
    }
    (out_dir / "ablation.json").write_text(json.dumps(summary, indent=2, default=float))
    _write_markdown(out_dir, summary)
    log.info("ablation -> %s", out_dir / "ablation.json")
    return summary
