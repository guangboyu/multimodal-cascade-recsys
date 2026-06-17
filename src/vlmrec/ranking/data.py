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


def build_ranking_data(paths: Paths, max_seq_len: int = 30) -> RankingData:
    d = load_retrieval_data(paths)
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
