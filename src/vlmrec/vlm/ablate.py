"""Feature-source ablation: what do VLM profiles add over CLIP image + raw-text embeddings?

For each source combo, trains the content two-tower AND the ranker in memory (no canonical
artifacts touched) and reports overall + cold-start metrics for both stages. The headline
question: do VLM profiles lift the cold-start slice, where content features matter most?
Writes ``data/vlm/ablation.json`` + ``WEEK8_RESULTS.md``.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from ..paths import Paths
from ..ranking.eval import _cold_users
from ..ranking.train import evaluate_ranking
from ..ranking.train import train as ranking_train
from ..retrieval.data import load_retrieval_data
from ..retrieval.train import evaluate as retrieval_evaluate
from ..retrieval.train import train as retrieval_train
from ..utils import get_logger, pick_device

log = get_logger("vlmrec.vlm.ablate")

COMBOS = [
    ("text", "image"),  # current production features (baseline)
    ("image",),
    ("vlm",),
    ("image", "vlm"),
    ("text", "image", "vlm"),
]


def _retrieval_cold(d):
    test_users = np.where(d.test_item >= 0)[0]
    tp = d.item_pop[d.test_item[test_users]]
    thr = int(np.quantile(tp, 0.25))
    return test_users[tp <= thr], thr


def run(cfg, paths: Paths) -> dict:
    device = pick_device(str(cfg.device))
    rr, rk = cfg.retrieval, cfg.ranking
    ks = tuple(int(k) for k in rr.ks)
    out_dir = paths.vlm
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict = {}
    for combo in COMBOS:
        name = "+".join(combo)
        log.info("=== ablation combo: %s ===", name)
        d = load_retrieval_data(paths, sources=combo)

        # retrieval: content two-tower, in-memory (save=False keeps canonical artifacts intact)
        m, model, d, T = retrieval_train(
            cfg,
            paths,
            feature_mode="content",
            epochs=int(rr.epochs),
            batch_size=int(rr.batch_size),
            lr=float(rr.lr),
            out_dim=int(rr.out_dim),
            hidden=tuple(rr.hidden),
            temperature=float(rr.temperature),
            eval_every=int(rr.eval_every),
            ks=ks,
            seed=int(cfg.seed),
            d=d,
            save=False,
            return_model=True,
        )
        cold_users, thr = _retrieval_cold(d)
        cold = retrieval_evaluate(model, d, T, "test", ks, user_subset=cold_users)
        del model, T
        torch.cuda.empty_cache()

        # ranking: standard week-3-style ranker on the same feature combo (no hard negs/score —
        # this isolates the content representation, not the cascade machinery)
        from ..ranking.data import build_ranking_data

        prebuilt = build_ranking_data(paths, max_seq_len=int(rk.max_seq_len), base=d)
        rmodel, rd, (content, cat, seq_t, ret_ui), rmetrics = ranking_train(
            cfg,
            paths,
            rd=prebuilt,
            label=f"ablate:{name}",
            epochs=int(rk.epochs),
            batch_size=int(rk.batch_size),
            lr=float(rk.lr),
            n_neg=int(rk.n_neg_train),
            n_neg_eval=int(rk.n_neg_eval),
            eval_user_sample=rk.get("eval_user_sample"),
            d_model=int(rk.d_model),
            n_tasks=int(rk.n_tasks),
            max_seq_len=int(rk.max_seq_len),
            seed=int(cfg.seed),
            sources=combo,
        )
        rcold, _ = _cold_users(rd.base)
        rcold_m = evaluate_ranking(
            rmodel,
            rd,
            content,
            cat,
            seq_t,
            device,
            n_neg=int(rk.n_neg_eval),
            user_subset=rcold,
            ret_ui=ret_ui,
            max_users=rk.get("eval_user_sample"),
        )
        del rmodel, content, cat, seq_t
        torch.cuda.empty_cache()

        results[name] = {
            "content_dim": int(d.content_dim),
            "retrieval": {
                "test": m["test"],
                "cold_Recall@100": cold["Recall@100"],
                "cold_threshold_pop": thr,
            },
            "ranking": {
                "click_GAUC": rmetrics["click_GAUC"],
                "cold_GAUC": rcold_m["click_GAUC"],
            },
        }
        log.info(
            "[%s] R@100=%.4f cold=%.4f | GAUC=%.4f cold=%.4f",
            name,
            m["test"]["Recall@100"],
            cold["Recall@100"],
            rmetrics["click_GAUC"],
            rcold_m["click_GAUC"],
        )

    summary = {"combos": results}
    (out_dir / "ablation.json").write_text(json.dumps(summary, indent=2, default=float))
    _write_md(out_dir, results)
    log.info("vlm ablation -> %s", out_dir / "ablation.json")
    return summary


def _write_md(out_dir, results: dict) -> None:
    lines = [
        "# Week 8 — VLM item-profile ablation",
        "",
        "Feature-source combos through the SAME two-tower + ranker recipes (test split; cold =",
        "bottom-quartile target train-popularity).",
        "",
        "| sources | dim | R@100 | R@100 (cold) | GAUC | GAUC (cold) |",
        "|---|---|---|---|---|---|",
    ]
    for name, r in results.items():
        lines.append(
            f"| {name} | {r['content_dim']} | {r['retrieval']['test']['Recall@100']:.4f} "
            f"| {r['retrieval']['cold_Recall@100']:.4f} | {r['ranking']['click_GAUC']:.4f} "
            f"| {r['ranking']['cold_GAUC']:.4f} |"
        )
    (out_dir / "WEEK8_RESULTS.md").write_text("\n".join(lines) + "\n")
