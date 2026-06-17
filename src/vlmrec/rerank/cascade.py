"""Week-4 orchestrator: hard-negative cascade fix + pre-rank consistency + diversity tradeoff.

1. precompute the retriever's candidates,
2. retrain the ranker on HARD negatives (sampled from those candidates) — fixing the Week-3 cascade,
3. distill a lightweight pre-ranker from it (consistency check),
4. measure the relevance↔diversity tradeoff (MMR λ sweep + DPP) on the pre-ranked list.
"""

from __future__ import annotations

import json
import math

import numpy as np
import torch

from ..paths import Paths
from ..ranking.eval import _cascade_eval
from ..ranking.train import train as ranking_train
from ..utils import get_logger, pick_device
from .candidates import candidates_path, precompute_candidates
from .postprocess import dpp_greedy, intra_list_diversity, mmr
from .prerank import distill_prerank

log = get_logger("vlmrec.rerank.cascade")


def _load_naive_cascade(paths: Paths):
    try:
        return json.loads((paths.data / "ranking" / "ablation.json").read_text()).get("cascade")
    except Exception:  # noqa: BLE001
        return None


def _ndcg_at(order, tgt_local, k) -> float:
    if len(tgt_local) == 0:
        return 0.0
    pos = np.where(order == int(tgt_local[0]))[0]
    if len(pos) == 0 or pos[0] >= k:
        return 0.0
    return 1.0 / math.log2(pos[0] + 2)


@torch.no_grad()
def _diversity_tradeoff(
    model,
    rd,
    tensors,
    cand,
    paths,
    device,
    sample=1500,
    prerank_m=50,
    k=10,
    lambdas=(1.0, 0.7, 0.5),
):
    content, cat, seq_t = tensors
    d = rd.base
    item_e = np.load(paths.data / "retrieval" / "item_emb_content.npy")  # (N, 128) normalized
    big_k = cand.shape[1]
    test_users = np.where(d.test_item >= 0)[0]
    rng = np.random.default_rng(0)
    users = rng.choice(test_users, size=min(sample, len(test_users)), replace=False)
    cand_t = torch.tensor(cand, device=device)
    methods = [f"mmr@{lam}" for lam in lambdas] + ["dpp"]
    agg = {m: {"ndcg": [], "div": []} for m in methods}

    for s in range(0, len(users), 256):
        ub = users[s : s + 256]
        cand_flat = cand_t[torch.as_tensor(ub, device=device)].reshape(-1)
        seq_b = seq_t[torch.as_tensor(np.repeat(ub, big_k), device=device)]
        sc = (
            torch.sigmoid(model(cand_flat, seq_b, content, cat)[:, 0])
            .reshape(len(ub), big_k)
            .cpu()
            .numpy()
        )
        for r in range(len(ub)):
            cset = cand[ub[r]]
            top = np.argsort(-sc[r])[:prerank_m]  # pre-rank cut to M by relevance
            scores, items, emb = sc[r][top], cset[top], item_e[cset[top]]
            tgt_local = np.where(items == int(d.test_item[ub[r]]))[0]
            for lam in lambdas:
                order = mmr(scores, emb, k, lam)
                agg[f"mmr@{lam}"]["ndcg"].append(_ndcg_at(order, tgt_local, k))
                agg[f"mmr@{lam}"]["div"].append(intra_list_diversity(emb[order]))
            order = dpp_greedy(scores, emb, k)
            agg["dpp"]["ndcg"].append(_ndcg_at(order, tgt_local, k))
            agg["dpp"]["div"].append(intra_list_diversity(emb[order]))

    return {
        m: {
            f"ndcg@{k}": round(float(np.mean(v["ndcg"])), 5),
            f"diversity@{k}": round(float(np.mean(v["div"])), 5),
        }
        for m, v in agg.items()
    }


def _write_md(out_dir, s) -> None:
    n, h = s["cascade_naive"], s["cascade_hardneg"]
    lines = ["# Week 4 — Pre-ranking + post-processing", ""]
    if n and h:
        nr, nk = n["ndcg@10_retrieval_order"], n["ndcg@10_ranker_order"]
        hr, hk = h["ndcg@10_retrieval_order"], h["ndcg@10_ranker_order"]
        lines += [
            "## Cascade diagnosis (honest negative result)",
            "NDCG@10 when re-ranking the retriever's candidates (retrieval order -> ranker order):",
            "",
            "| ranker | retrieval order | ranker order |",
            "|---|---|---|",
            f"| naive (random negatives) | {nr:.3f} | {nk:.3f} |",
            f"| hard negatives | {hr:.3f} | {hk:.3f} |",
            "",
            "Both rankers *lower* NDCG vs retrieval order; naive hard negatives make it worse.",
            "Labelling retrieved items as negatives teaches 'retrieved = negative', penalizing the",
            "held-out positive (itself retrieved); the metric is retrieval-favoring. Principled",
            "fix: feed the retrieval/pre-rank score to the ranker as a feature so it refines (not",
            "discards) the retrieval signal (see docs/PITFALLS.md).",
            "",
        ]
    if s.get("prerank_consistency"):
        rec = next(iter(s["prerank_consistency"].values()))
        lines += [
            "## Pre-ranking (distilled)",
            f"- the lightweight pre-ranker's top-50 recovers **{rec:.1%}** of the ranker's top-10 "
            "(a cheap, consistent candidate cut).",
            "",
        ]
    if s.get("diversity_tradeoff"):
        lines += [
            "## Post-processing — relevance vs diversity (on the pre-ranked top-50)",
            "",
            "| method | NDCG | diversity |",
            "|---|---|---|",
        ]
        for m, v in s["diversity_tradeoff"].items():
            nd = next(val for key, val in v.items() if key.startswith("ndcg"))
            dv = next(val for key, val in v.items() if key.startswith("diversity"))
            lines.append(f"| {m} | {nd:.4f} | {dv:.4f} |")
        lines.append("")
        lines.append("Lower MMR λ (and DPP) trade a little relevance for a more diverse list.")
    (out_dir / "WEEK4_RESULTS.md").write_text("\n".join(lines) + "\n")


def run(cfg, paths: Paths) -> dict:
    device = pick_device(str(cfg.device))
    r, rk = cfg.rerank, cfg.ranking
    out_dir = paths.data / "rerank"
    out_dir.mkdir(parents=True, exist_ok=True)

    cpath = candidates_path(paths)
    cand = np.load(cpath) if cpath.exists() else precompute_candidates(cfg, paths, k=int(r.cand_k))

    log.info("=== training ranker on HARD negatives (frac=%.2f) ===", float(r.hard_neg_frac))
    model, rd, tensors, metrics = ranking_train(
        cfg,
        paths,
        label="hardneg",
        epochs=int(rk.epochs),
        batch_size=int(rk.batch_size),
        lr=float(rk.lr),
        n_neg=int(rk.n_neg_train),
        n_neg_eval=int(rk.n_neg_eval),
        d_model=int(rk.d_model),
        n_tasks=int(rk.n_tasks),
        max_seq_len=int(rk.max_seq_len),
        seed=int(cfg.seed),
        cand_topk=cand,
        hard_neg_frac=float(r.hard_neg_frac),
    )
    torch.save(model.state_dict(), out_dir / "ranker_hardneg.pt")

    cascade_hardneg = _cascade_eval(cfg, paths, (model, rd, tensors), device)
    log.info("cascade (hard-neg ranker): %s", cascade_hardneg)

    student, consistency = distill_prerank(
        model, rd, tensors, cand, device, epochs=int(r.prerank_epochs)
    )
    torch.save(student.state_dict(), out_dir / "prerank.pt")

    diversity = _diversity_tradeoff(model, rd, tensors, cand, paths, device, k=int(r.final_k))
    log.info("diversity tradeoff: %s", diversity)

    summary = {
        "cascade_naive": _load_naive_cascade(paths),
        "cascade_hardneg": cascade_hardneg,
        "ranker_hardneg_GAUC": metrics,
        "prerank_consistency": consistency,
        "diversity_tradeoff": diversity,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    _write_md(out_dir, summary)
    log.info("week4 results -> %s", out_dir / "results.json")
    return summary
