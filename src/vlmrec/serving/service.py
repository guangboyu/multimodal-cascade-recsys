"""The serving cascade: retrieve -> pre-rank -> rank -> post-process, with per-stage latency.

Mirrors the offline stages: FAISS ANN retrieval (mask seen) -> lightweight pre-ranker cut ->
heavy DIN/DCN/MMoE ranker -> score fusion + diversity (MMR/DPP). Returns the final item list and a
per-stage millisecond breakdown so a latency budget can be tracked.
"""

from __future__ import annotations

import time

import numpy as np
import torch

from ..rerank.postprocess import dpp_greedy, fuse_scores, mmr
from .registry import Registry


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


@torch.no_grad()
def _score(model, reg: Registry, user_idx: int, cand: np.ndarray) -> np.ndarray:
    cand_t = torch.tensor(cand)
    seq_b = reg.seq_t[torch.tensor([int(user_idx)] * len(cand))]
    return model(cand_t, seq_b, reg.content, reg.cat).numpy()


@torch.no_grad()
def recommend(
    reg: Registry,
    user_idx: int,
    k_retrieve=200,
    k_prerank=50,
    k_final=20,
    diversity="mmr",
    mmr_lambda=0.7,
) -> dict:
    d = reg.d
    lat: dict[str, float] = {}

    # 1. retrieval (FAISS ANN), drop already-seen items
    t0 = time.perf_counter()
    ue = reg.tt.user_embeddings_eval(
        torch.tensor([int(user_idx)]), reg.user_sum, reg.user_count, reg.content[:-1]
    )
    _, idx = reg.index.search(ue.numpy().astype("float32"), k_retrieve + 64)
    seen = set(d.seen_items[d.seen_indptr[user_idx] : d.seen_indptr[user_idx + 1]].tolist())
    cand = np.array([c for c in idx[0] if c not in seen][:k_retrieve], dtype=np.int64)
    lat["retrieve_ms"] = (time.perf_counter() - t0) * 1000

    # 2. pre-rank (lightweight) cut to k_prerank
    t0 = time.perf_counter()
    if reg.prerank is not None and len(cand) > k_prerank:
        ps = _score(reg.prerank, reg, user_idx, cand)[:, 0]
        cand = cand[np.argsort(-ps)[:k_prerank]]
    lat["prerank_ms"] = (time.perf_counter() - t0) * 1000

    # 3. rank (heavy DIN + DCN-v2 + MMoE)
    t0 = time.perf_counter()
    logits = _score(reg.ranker, reg, user_idx, cand)
    pclick = _sigmoid(logits[:, 0])
    psat = _sigmoid(logits[:, 1]) if logits.shape[1] > 1 else np.ones_like(pclick)
    lat["rank_ms"] = (time.perf_counter() - t0) * 1000

    # 4. post-process: fuse + diversity
    t0 = time.perf_counter()
    score = fuse_scores(pclick, psat)
    emb = reg.item_e[cand]
    if diversity == "mmr":
        order = mmr(score, emb, k_final, mmr_lambda)
    elif diversity == "dpp":
        order = dpp_greedy(score, emb, k_final)
    else:
        order = np.argsort(-score)[:k_final]
    final = cand[order]
    lat["postprocess_ms"] = (time.perf_counter() - t0) * 1000

    lat["total_ms"] = round(sum(lat.values()), 2)
    return {
        "user_idx": int(user_idx),
        "items": [int(x) for x in final],
        "scores": [round(float(score[o]), 4) for o in order],
        "latency_ms": {k: round(v, 2) for k, v in lat.items()},
    }
