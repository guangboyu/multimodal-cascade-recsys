"""FAISS ANN index over item-tower embeddings — the production retrieval path.

Embeddings are L2-normalized, so inner-product search == cosine similarity. For this catalog
(~25k items) a flat IP index is exact and instant; HNSW/IVF are wired in to show the
approximate-NN speed/recall trade-off used at real scale. `ann_vs_exact_recall` quantifies the
recall lost to approximation.
"""

from __future__ import annotations

import numpy as np


def build_index(
    item_emb: np.ndarray,
    kind: str = "flat",
    nlist: int = 256,
    hnsw_m: int = 32,
    nprobe: int = 8,
    ef_search: int = 256,
):
    import faiss

    x = np.ascontiguousarray(item_emb.astype(np.float32))
    dim = x.shape[1]
    if kind == "flat":
        index = faiss.IndexFlatIP(dim)
    elif kind == "hnsw":
        index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        # default efSearch (16) can't fill k>16 requests — unfilled slots come back as -1
        index.hnsw.efSearch = ef_search
    elif kind == "ivf":
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dim), dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(x)
        index.nprobe = nprobe
    else:
        raise ValueError(f"unknown index kind: {kind}")
    index.add(x)
    return index


def query(index, user_emb: np.ndarray, k: int):
    q = np.ascontiguousarray(user_emb.astype(np.float32))
    scores, idx = index.search(q, k)
    return idx, scores


def ann_vs_exact_recall(
    item_emb: np.ndarray, user_emb: np.ndarray, k: int = 100, kind: str = "hnsw"
) -> float:
    """Fraction of the exact top-k recovered by the ANN index (averaged over query users)."""
    idx, _ = query(build_index(item_emb, kind=kind), user_emb, k)
    exact = np.argpartition(-(user_emb @ item_emb.T), k, axis=1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(idx, exact, strict=False)]))
