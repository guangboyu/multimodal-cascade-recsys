"""Item2item co-visitation — a content-free, collaborative retrieval source for the long tail.

For items co-occurring in a user's train history, accumulate co-occurrence weight normalized by
sqrt(pop_i * pop_j) (damps blockbusters). At query time we score candidates by summing neighbor
weights across the user's history. This complements the two-tower (the IDEA_REPORT's "one i2i
source") and is a strong, simple baseline.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from .data import RetrievalData


def build_covisitation(d: RetrievalData, top_n: int = 50, max_history: int = 50) -> dict:
    co: dict[int, dict[int, float]] = defaultdict(lambda: defaultdict(float))
    indptr, items = d.seen_indptr, d.seen_items
    for u in range(d.n_users):
        hist = items[indptr[u] : indptr[u + 1]]
        if len(hist) > max_history:
            hist = hist[:max_history]
        for x in range(len(hist)):
            ix = int(hist[x])
            for y in range(x + 1, len(hist)):
                jy = int(hist[y])
                co[ix][jy] += 1.0
                co[jy][ix] += 1.0

    pop = d.item_pop.astype(np.float64)
    neigh: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for i, nbrs in co.items():
        js = np.fromiter(nbrs.keys(), dtype=np.int64)
        ws = np.fromiter(nbrs.values(), dtype=np.float64)
        ws = ws / np.sqrt((pop[i] + 1.0) * (pop[js] + 1.0))  # popularity damping
        order = np.argsort(-ws)[:top_n]
        neigh[i] = (js[order], ws[order])
    return neigh


def evaluate_i2i(
    d: RetrievalData, neigh: dict, split: str = "test", ks=(10, 50, 100, 500), user_subset=None
) -> dict:
    targets = d.test_item if split == "test" else d.valid_item
    if user_subset is None:
        users = np.where(targets >= 0)[0]
    else:
        users = np.asarray(user_subset, dtype=np.int64)
        users = users[targets[users] >= 0]
    indptr, items = d.seen_indptr, d.seen_items
    max_k = max(ks)
    hits = {k: 0 for k in ks}
    ndcg = {k: 0.0 for k in ks}

    for u in users:
        hist = items[indptr[u] : indptr[u + 1]]
        scores: dict[int, float] = defaultdict(float)
        for h in hist:
            nb = neigh.get(int(h))
            if nb is None:
                continue
            for j, w in zip(nb[0], nb[1], strict=False):
                scores[int(j)] += w
        for h in hist:  # never recommend already-seen
            scores.pop(int(h), None)
        if not scores:
            continue
        ranked = sorted(scores, key=scores.get, reverse=True)[:max_k]
        tgt = int(targets[u])
        if tgt in ranked:
            rank = ranked.index(tgt)
            for k in ks:
                if rank < k:
                    hits[k] += 1
                    ndcg[k] += 1.0 / np.log2(rank + 2)

    n = len(users)
    out = {f"Recall@{k}": round(hits[k] / n, 5) for k in ks}
    out |= {f"NDCG@{k}": round(ndcg[k] / n, 5) for k in ks}
    out["n_users_eval"] = int(n)
    return out
