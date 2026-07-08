"""Ranking data: behavior sequences (for DIN), multi-task labels, and negative sampling.

Reuses the retrieval data layer (content / cat / splits / seen-history / popularity) and adds:
  * per-user **padded behavior sequences** (last L train items, time-ordered) for DIN
  * training **positives** carrying a satisfaction label (rating >= 4)
  * negatives are sampled on the fly in the training loop (random items)

Sequence pad = ``n_items`` (a zero content row is appended so gathering pad is safe).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl

from ..paths import Paths
from ..retrieval.data import RetrievalData, load_retrieval_data
from ..utils import get_logger

log = get_logger("vlmrec.ranking.data")


@dataclass
class RankingData:
    base: RetrievalData
    seq: np.ndarray  # (U, L) int64, pad == n_items
    seq_len: np.ndarray  # (U,) int64
    pos: np.ndarray  # (P, 3) int64: [user_idx, item_idx, satisfaction]
    seq_len_max: int

    @property
    def n_users(self) -> int:
        return self.base.n_users

    @property
    def n_items(self) -> int:
        return self.base.n_items

    @property
    def pad_idx(self) -> int:
        return self.base.n_items


def sample_negatives(
    rng: np.random.Generator,
    pu: np.ndarray,
    pi: np.ndarray,
    n_items: int,
    n_neg: int,
    valid_item: np.ndarray,
    test_item: np.ndarray,
    cand_topk: np.ndarray | None = None,
    hard_frac: float = 0.0,
    max_redraws: int = 3,
    hard_top: int | None = None,
) -> np.ndarray:
    """Sample (B, n_neg) negatives for positives (pu, pi): ``hard_frac`` of them from each user's
    retrieval candidates, the rest uniform over the catalog.

    Excludes the row's own positive and the user's held-out valid/test items from BOTH pools —
    a user's future positive is usually retrieved, so labelling it click=0 poisons hard-negative
    training (the Week-4 cascade failure; see docs/PITFALLS.md). Collisions are rejection-resampled
    (hard slots redraw from the candidate row, random slots from the catalog); the final fallback
    walks stragglers off the banned set deterministically.
    """
    b = len(pu)
    n_hard = int(round(n_neg * hard_frac)) if cand_topk is not None else 0
    # hard_top limits hard draws to the head of the retrieval list — the confusable candidates
    k_pool = min(hard_top or cand_topk.shape[1], cand_topk.shape[1]) if n_hard else 0
    parts = []
    if n_hard > 0:
        cols = rng.integers(0, k_pool, size=(b, n_hard))
        parts.append(cand_topk[pu[:, None], cols])
    if n_neg - n_hard > 0:
        parts.append(rng.integers(0, n_items, size=(b, n_neg - n_hard)))
    negs = np.concatenate(parts, axis=1)

    banned = np.stack([pi, valid_item[pu], test_item[pu]], axis=1)  # (B, 3); -1 never collides
    for _ in range(max_redraws):
        bad = (negs[:, :, None] == banned[:, None, :]).any(axis=2)
        if not bad.any():
            return negs
        rows, slots = np.nonzero(bad)
        hard_slot = slots < n_hard
        if hard_slot.any():
            hr = rows[hard_slot]
            cols = rng.integers(0, k_pool, size=len(hr))
            negs[hr, slots[hard_slot]] = cand_topk[pu[hr], cols]
        if (~hard_slot).any():
            negs[rows[~hard_slot], slots[~hard_slot]] = rng.integers(
                0, n_items, size=int((~hard_slot).sum())
            )
    # guarantee: banned has <=3 distinct values per row, so +1 steps terminate in <=4 passes
    bad = (negs[:, :, None] == banned[:, None, :]).any(axis=2)
    while bad.any():
        idx = np.nonzero(bad)
        negs[idx] = (negs[idx] + 1) % n_items
        bad = (negs[:, :, None] == banned[:, None, :]).any(axis=2)
    return negs


def build_ranking_data(
    paths: Paths,
    max_seq_len: int = 30,
    sources=("text", "image"),
    base: RetrievalData | None = None,
) -> RankingData:
    d = base if base is not None else load_retrieval_data(paths, sources=sources)
    n_users, n_items = d.n_users, d.n_items

    # per-user time-ordered train sequence (seen CSR is already (user, time)-ordered), last L
    seq = np.full((n_users, max_seq_len), n_items, dtype=np.int64)  # pad = n_items
    seq_len = np.zeros(n_users, dtype=np.int64)
    indptr, items = d.seen_indptr, d.seen_items
    for u in range(n_users):
        h = items[indptr[u] : indptr[u + 1]][-max_seq_len:]
        seq[u, : len(h)] = h
        seq_len[u] = len(h)

    ix = pl.read_parquet(paths.interactions_parquet).filter(pl.col("split") == "train")
    pu = ix.get_column("user_idx").to_numpy().astype(np.int64)
    pi = ix.get_column("item_idx").to_numpy().astype(np.int64)
    ps = ix.get_column("strong").to_numpy().astype(np.int64)
    pos = np.stack([pu, pi, ps], axis=1)

    log.info(
        "ranking data: users=%s items=%s positives=%s seq_len_max=%s",
        f"{n_users:,}",
        f"{n_items:,}",
        f"{len(pos):,}",
        max_seq_len,
    )
    return RankingData(base=d, seq=seq, seq_len=seq_len, pos=pos, seq_len_max=max_seq_len)
