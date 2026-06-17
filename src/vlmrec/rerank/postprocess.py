"""Post-processing: score fusion + diversity re-ranking (MMR, greedy-MAP DPP) + business rules.

Operates on a single user's candidate list: relevance scores + item content embeddings. Diversity
re-ranking trades a little relevance for a less redundant list (the gap between an "accurate" and a
"good" list). Includes intra-list diversity / category-entropy metrics for the relevance↔diversity
tradeoff curve.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np


def fuse_scores(p_click: np.ndarray, p_sat: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """Business value ≈ P(click) · P(satisfaction)^alpha."""
    return p_click * (p_sat**alpha)


def mmr(scores: np.ndarray, emb: np.ndarray, k: int, lam: float = 0.7) -> np.ndarray:
    """Maximal Marginal Relevance: lam·relevance − (1−lam)·max-similarity-to-selected."""
    n = len(scores)
    sim = emb @ emb.T
    selected: list[int] = []
    remaining = set(range(n))
    while len(selected) < min(k, n):
        if not selected:
            j = int(np.argmax(scores))
        else:
            best, best_val = -1, -1e18
            for c in remaining:
                val = lam * scores[c] - (1 - lam) * max(sim[c, s] for s in selected)
                if val > best_val:
                    best_val, best = val, c
            j = best
        selected.append(j)
        remaining.discard(j)
    return np.array(selected)


def dpp_greedy(scores: np.ndarray, emb: np.ndarray, k: int) -> np.ndarray:
    """Greedy MAP inference for a DPP with kernel L = diag(q)·S·diag(q) (Chen et al., 2018)."""
    n = len(scores)
    q = np.exp(0.5 * (scores - scores.max()))  # quality
    sim = emb @ emb.T
    L = (q[:, None] * sim) * q[None, :]
    cis = np.zeros((min(k, n), n))
    d2 = np.diag(L).copy()
    selected: list[int] = []
    for it in range(min(k, n)):
        j = int(np.argmax(d2))
        selected.append(j)
        if it == n - 1 or d2[j] <= 0:
            break
        ci = (L[j] - cis[:it].T @ cis[:it, j]) / np.sqrt(d2[j])
        cis[it] = ci
        d2 = d2 - ci**2
        d2[selected] = -np.inf
    return np.array(selected)


def category_cap(order: np.ndarray, categories: np.ndarray, cap: int) -> np.ndarray:
    """Business rule: at most `cap` items per category, preserving order."""
    cnt: dict = defaultdict(int)
    out = []
    for i in order:
        c = int(categories[i])
        if cnt[c] < cap:
            out.append(i)
            cnt[c] += 1
    return np.array(out)


def intra_list_diversity(emb_subset: np.ndarray) -> float:
    """1 − mean pairwise cosine similarity over the selected list (higher = more diverse)."""
    n = len(emb_subset)
    if n < 2:
        return 0.0
    sim = emb_subset @ emb_subset.T
    off_mean = (sim.sum() - np.trace(sim)) / (n * (n - 1))
    return float(1 - off_mean)
