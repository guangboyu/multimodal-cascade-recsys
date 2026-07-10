"""Rerank orchestrator: the cascade-consistency fix + pre-rank consistency + diversity tradeoff.

1. precompute the retriever's candidates (+ scores + user embeddings),
2. retrain the ranker on CLEAN hard negatives (held-out positives excluded) — two variants:
   without and with the retrieval-score feature, isolating each repair's contribution,
3. tune the retrieval/ranker score-fusion alpha on the valid split, report on test,
4. distill a lightweight pre-ranker from the fixed ranker (consistency check),
5. measure the relevance↔diversity tradeoff (MMR λ sweep + DPP) on the pre-ranked list.
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
from .candidates import candidates_path, precompute_candidates, user_emb_path
from .postprocess import dpp_greedy, intra_list_diversity, mmr
from .prerank import distill_prerank

log = get_logger("vlmrec.rerank.cascade")

FUSION_ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)


def _load_naive_cascade(paths: Paths):
    try:
        return json.loads((paths.data / "ranking" / "ablation.json").read_text()).get("cascade")
    except Exception:  # noqa: BLE001
        return None


def _load_historical(paths: Paths, key: str):
    """Preserve a metric from the previous results.json (e.g. the poisoned hard-neg run)."""
    try:
        return json.loads((paths.data / "rerank" / "results.json").read_text()).get(key)
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
    content, cat, seq_t, ret_ui = tensors
    use_score = bool(getattr(model, "use_retrieval_score", False))
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
        users_t = torch.as_tensor(np.repeat(ub, big_k), device=device)
        cand_flat = cand_t[torch.as_tensor(ub, device=device)].reshape(-1)
        seq_b = seq_t[users_t]
        ret = (ret_ui[0][users_t] * ret_ui[1][cand_flat]).sum(-1) if use_score else None
        sc = (
            torch.sigmoid(model(cand_flat, seq_b, content, cat, ret_score=ret)[:, 0])
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


VARIANT_LABELS = {
    "hardneg_clean": "hard negatives, clean pool (pointwise BCE)",
    "scorefeat_bce": "+ retrieval-score feature (pointwise BCE)",
    "scorefeat_softmax": "+ listwise softmax over the slate",
    "scorefeat_softmax_slate": "+ serving-like slate (16 negs, 75% hard from top-50)",
}


def _write_md(out_dir, s) -> None:
    lines = ["# Week 4/7 — Cascade fix + pre-ranking + post-processing", ""]
    rows = [
        ("naive (random negatives)", s.get("cascade_naive")),
        ("hard negatives (poisoned: held-out positives labelled 0)", s.get("cascade_hardneg")),
    ]
    for name, v in (s.get("variants") or {}).items():
        rows.append((VARIANT_LABELS.get(name, name), v["test"]))
    lines += [
        "## Cascade: NDCG@10 re-ranking the retriever's top-200 (test)",
        "",
        "| ranker | retrieval order | ranker order |",
        "|---|---|---|",
    ]
    for name, c in rows:
        if c:
            r0, k0 = c["ndcg@10_retrieval_order"], c["ndcg@10_ranker_order"]
            lines.append(f"| {name} | {r0:.3f} | {k0:.3f} |")
    if s.get("best_variant"):
        lines += ["", f"Serving checkpoint = **{s['best_variant']}** (chosen on valid, not test)."]
    if s.get("fusion"):
        f = s["fusion"]
        lines += [
            "",
            f"Score fusion α·retrieval + (1−α)·ranker: α={f['alpha']} (tuned on valid) → "
            f"test NDCG@10 **{f['test_ndcg@10']:.3f}**.",
        ]
    lines += [
        "",
        "The failure modes and the fix are documented in docs/PITFALLS.md:",
        "random negatives never teach the serving distribution; naive hard negatives poison",
        "labels with held-out positives; the fix = clean negative pool + the retrieval score",
        "as a ranker input so ranking refines (not fights) the retrieval signal.",
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
    if cpath.exists() and user_emb_path(paths).exists():
        cand = np.load(cpath)
    else:  # (re)compute — older runs lack the score/user-embedding artifacts
        cand = precompute_candidates(cfg, paths, k=int(r.cand_k))

    from ..retrieval.data import cfg_sources

    common = dict(
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
        cand_topk=cand,
        hard_neg_frac=float(r.hard_neg_frac),
        sources=cfg_sources(cfg),
    )

    # variant grid: each row isolates one repair (negative hygiene -> score feature -> listwise
    # loss -> serving-like slate). Selection on VALID cascade NDCG; test is only reported.
    variants = [
        ("hardneg_clean", dict()),
        ("scorefeat_bce", dict(use_retrieval_score=True)),
        ("scorefeat_softmax", dict(use_retrieval_score=True, loss_type="softmax")),
        (
            # slate that looks like serving: big, mostly-hard, drawn from the confusable head
            "scorefeat_softmax_slate",
            dict(
                use_retrieval_score=True,
                loss_type="softmax",
                n_neg=16,
                hard_neg_frac=0.75,
                hard_neg_top=50,
            ),
        ),
    ]
    per_variant: dict[str, dict] = {}
    best = None  # (valid_ndcg, name, model, rd, tensors, metrics)
    for name, flags in variants:
        log.info("=== ranker variant: %s ===", name)
        model, rd, tensors, metrics = ranking_train(cfg, paths, label=name, **(common | flags))
        valid = _cascade_eval(cfg, paths, (model, rd, tensors), device, target="valid")
        test = _cascade_eval(cfg, paths, (model, rd, tensors), device, target="test")
        per_variant[name] = {"GAUC": metrics, "valid": valid, "test": test}
        log.info("[%s] valid ndcg=%s test=%s", name, valid["ndcg@10_ranker_order"], test)
        score = valid["ndcg@10_ranker_order"]
        if best is None or score > best[0]:
            best = (score, name, model, rd, tensors, metrics)
        else:
            del model, tensors
            torch.cuda.empty_cache()

    _, best_name, model, rd, tensors, metrics = best
    use_score = bool(getattr(model, "use_retrieval_score", False))
    log.info("=== best variant on valid: %s (score_feature=%s) ===", best_name, use_score)
    torch.save(model.state_dict(), out_dir / "ranker_cascade.pt")
    (out_dir / "ranker_cascade.json").write_text(
        json.dumps(
            {
                "variant": best_name,
                "use_retrieval_score": use_score,
                "d_model": int(rk.d_model),
                "n_tasks": int(rk.n_tasks),
            }
        )
    )
    cascade_scorefeat = per_variant[best_name]["test"]

    # score fusion: tune alpha on the VALID split, report the chosen alpha on test
    fuse_valid = _cascade_eval(
        cfg, paths, (model, rd, tensors), device, target="valid", alphas=FUSION_ALPHAS
    )
    best_alpha = max(fuse_valid["ndcg@10_fused"], key=fuse_valid["ndcg@10_fused"].get)
    fuse_test = _cascade_eval(
        cfg, paths, (model, rd, tensors), device, target="test", alphas=[float(best_alpha)]
    )
    fusion = {
        "alpha": float(best_alpha),
        "valid_sweep": fuse_valid["ndcg@10_fused"],
        "test_ndcg@10": fuse_test["ndcg@10_fused"][best_alpha],
    }
    log.info("fusion: %s", fusion)

    student, consistency = distill_prerank(
        model, rd, tensors, cand, device, epochs=int(r.prerank_epochs)
    )
    torch.save(student.state_dict(), out_dir / "prerank.pt")
    (out_dir / "prerank.json").write_text(
        json.dumps(
            {
                # must mirror the distilled student's ACTUAL architecture — the student inherits
                # the winning teacher's score-feature flag, which is not guaranteed True
                "use_retrieval_score": bool(getattr(student, "use_retrieval_score", False)),
                "d_model": 64,
                "n_tasks": 1,
            }
        )
    )

    diversity = _diversity_tradeoff(model, rd, tensors, cand, paths, device, k=int(r.final_k))
    log.info("diversity tradeoff: %s", diversity)

    summary = {
        "cascade_naive": _load_naive_cascade(paths),
        "cascade_hardneg": _load_historical(paths, "cascade_hardneg"),  # poisoned run, kept
        "variants": per_variant,
        "best_variant": best_name,
        "cascade_scorefeat": cascade_scorefeat,
        "fusion": fusion,
        "ranker_cascade_GAUC": metrics,
        "prerank_consistency": consistency,
        "diversity_tradeoff": diversity,
    }
    (out_dir / "results.json").write_text(json.dumps(summary, indent=2, default=float))
    _write_md(out_dir, summary)
    log.info("cascade results -> %s", out_dir / "results.json")
    return summary
