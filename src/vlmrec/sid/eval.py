"""Semantic-ID ablation: do RQ-VAE codes beat raw item IDs, especially on cold-start?

Two-tower modes (id / sid / content / content_sid) and the ranker (± SID feature) trained with
the same recipes, evaluated overall + cold-start. The headline contrast: ``sid`` vs ``id`` — the
same collaborative user vector, but the item pointer replaced by composed semantic codes that
cold items share with warm ones. Writes ``data/sid/ablation.json`` + ``WEEK9_RESULTS.md``.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from ..paths import Paths
from ..ranking.eval import _cold_users
from ..ranking.train import evaluate_ranking
from ..ranking.train import train as ranking_train
from ..retrieval.data import cfg_sources, load_retrieval_data
from ..retrieval.train import evaluate as retrieval_evaluate
from ..retrieval.train import train as retrieval_train
from ..utils import get_logger, pick_device
from .train import sid_codes_path

log = get_logger("vlmrec.sid.eval")

TOWER_MODES = ["id", "sid", "content", "content_sid"]


def run(cfg, paths: Paths) -> dict:
    device = pick_device(str(cfg.device))
    rr, rk = cfg.retrieval, cfg.ranking
    ks = tuple(int(k) for k in rr.ks)
    out_dir = paths.data / "sid"
    out_dir.mkdir(parents=True, exist_ok=True)
    sid_codes = np.load(sid_codes_path(paths))
    sources = cfg_sources(cfg)
    d = load_retrieval_data(paths, sources=sources)

    test_users = np.where(d.test_item >= 0)[0]
    tp = d.item_pop[d.test_item[test_users]]
    thr = int(np.quantile(tp, 0.25))
    cold_users = test_users[tp <= thr]

    towers: dict = {}
    for mode in TOWER_MODES:
        m, model, d, T = retrieval_train(
            cfg,
            paths,
            feature_mode=mode,
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
            sid_codes=sid_codes if "sid" in mode else None,
        )
        cold = retrieval_evaluate(model, d, T, "test", ks, user_subset=cold_users)
        towers[mode] = {"test": m["test"], "cold_Recall@100": cold["Recall@100"]}
        log.info(
            "[tower %s] R@100=%.4f cold=%.4f",
            mode,
            m["test"]["Recall@100"],
            cold["Recall@100"],
        )
        del model, T
        torch.cuda.empty_cache()

    rankers: dict = {}
    for name, use_sid in [("base", False), ("with_sid", True)]:
        rmodel, rd, (content, cat, seq_t, ret_ui), rmetrics = ranking_train(
            cfg,
            paths,
            label=f"sid:{name}",
            epochs=int(rk.epochs),
            batch_size=int(rk.batch_size),
            lr=float(rk.lr),
            n_neg=int(rk.n_neg_train),
            n_neg_eval=int(rk.n_neg_eval),
            d_model=int(rk.d_model),
            n_tasks=int(rk.n_tasks),
            max_seq_len=int(rk.max_seq_len),
            seed=int(cfg.seed),
            sources=sources,
            use_sid=use_sid,
            sid_codes=sid_codes if use_sid else None,
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
        )
        rankers[name] = {
            "click_GAUC": rmetrics["click_GAUC"],
            "cold_GAUC": rcold_m["click_GAUC"],
        }
        log.info(
            "[ranker %s] GAUC=%.4f cold=%.4f",
            name,
            rmetrics["click_GAUC"],
            rcold_m["click_GAUC"],
        )
        del rmodel, content, cat, seq_t
        torch.cuda.empty_cache()

    summary = {
        "sources": list(sources),
        "cold_threshold_pop": thr,
        "towers": towers,
        "rankers": rankers,
    }
    (out_dir / "ablation.json").write_text(json.dumps(summary, indent=2, default=float))
    _write_md(out_dir, summary)
    log.info("sid ablation -> %s", out_dir / "ablation.json")
    return summary


def _write_md(out_dir, s: dict) -> None:
    lines = [
        "# Week 9 — Semantic-ID ablation",
        "",
        "Same recipes, only the item representation changes (test split; cold = bottom-quartile",
        "target train-popularity).",
        "",
        "## Two-tower retrieval",
        "",
        "| item representation | R@100 | R@100 (cold) |",
        "|---|---|---|",
    ]
    for mode, r in s["towers"].items():
        lines.append(f"| {mode} | {r['test']['Recall@100']:.4f} | {r['cold_Recall@100']:.4f} |")
    lines += [
        "",
        "## Ranker (± SID feature)",
        "",
        "| ranker | GAUC | GAUC (cold) |",
        "|---|---|---|",
    ]
    for name, r in s["rankers"].items():
        lines.append(f"| {name} | {r['click_GAUC']:.4f} | {r['cold_GAUC']:.4f} |")
    (out_dir / "WEEK9_RESULTS.md").write_text("\n".join(lines) + "\n")
