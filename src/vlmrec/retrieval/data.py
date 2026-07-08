"""Retrieval data layer: assemble item features + user histories for two-tower train/eval.

Produces everything the two-tower needs, aligned to ``item_idx`` (row i == item_idx i):
  * ``content``        — dense multimodal item features: text(384) ⊕ image(512) ⊕ has_image(1)
  * ``cat_features``   — integer indices for embedding lookup: [item_id, price_bucket, category]
                         (the quantile-bucket + embedding pattern for structured features)
  * ``user_sum_content`` / ``user_count`` — per-user SUM and COUNT of train-history content,
                         so the user tower can mean-pool a history (leave-one-out at train time)
  * ``train_pairs``    — (user_idx, item_idx) training interactions (in-batch-negative source)
  * ``valid_item`` / ``test_item`` — per-user held-out targets (temporal leave-last-out)
  * ``seen_indptr`` / ``seen_items`` — CSR of train items per user, to mask already-seen at eval
  * ``item_pop``       — train popularity (popularity baseline + cold-start slicing)
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass

import numpy as np
import polars as pl

from ..paths import Paths
from ..utils import get_logger

log = get_logger("vlmrec.retrieval.data")


@dataclass
class RetrievalData:
    content: np.ndarray  # (N, Dc) float32
    cat_features: np.ndarray  # (N, n_cat) int64  [item_id, price_bucket, category_leaf]
    cat_cardinalities: list[int]
    cat_names: list[str]
    user_sum_content: np.ndarray  # (U, Dc) float32
    user_count: np.ndarray  # (U,) int64
    train_pairs: np.ndarray  # (P, 2) int64
    valid_item: np.ndarray  # (U,) int64 (-1 if none)
    test_item: np.ndarray  # (U,) int64 (-1 if none)
    seen_indptr: np.ndarray  # (U+1,) int64 — CSR row pointers into seen_items
    seen_items: np.ndarray  # (P,) int64 — train items, grouped by user
    item_pop: np.ndarray  # (N,) int64
    n_users: int
    n_items: int

    @property
    def content_dim(self) -> int:
        return int(self.content.shape[1])


def _parse_price(s) -> float:
    if s is None:
        return np.nan
    s = str(s).replace("$", "").replace(",", "").strip()
    if not s:
        return np.nan
    s = s.split()[0].split("-")[0]  # "9.99 - 19.99" -> "9.99"
    try:
        return float(s)
    except ValueError:
        return np.nan


def _quantile_buckets(values: np.ndarray, n_buckets: int) -> tuple[np.ndarray, int]:
    """Equal-frequency buckets; bucket 0 reserved for null/unknown. Returns (ids, cardinality)."""
    out = np.zeros(len(values), dtype=np.int64)  # 0 == null
    mask = ~np.isnan(values)
    if mask.sum() > 0:
        v = values[mask]
        edges = np.quantile(v, np.linspace(0, 1, n_buckets + 1)[1:-1])  # interior edges
        out[mask] = np.searchsorted(edges, v, side="right") + 1  # 1..n_buckets
    return out, n_buckets + 1


def _category_leaf_ids(cat_json: list, min_count: int) -> tuple[np.ndarray, int]:
    """Map the most-specific category (JSON list leaf) to an id; rare leaves -> 0 (OOV)."""
    leaves = []
    for c in cat_json:
        leaf = None
        try:
            arr = json.loads(c) if isinstance(c, str) else c
            if isinstance(arr, list) and arr:
                leaf = str(arr[-1])
        except (json.JSONDecodeError, TypeError):
            leaf = None
        leaves.append(leaf)
    counts = Counter(leaf for leaf in leaves if leaf)
    vocab = {leaf: i + 1 for i, leaf in enumerate(k for k, v in counts.items() if v >= min_count)}
    ids = np.array([vocab.get(leaf, 0) for leaf in leaves], dtype=np.int64)
    return ids, len(vocab) + 1


SOURCE_ORDER = ("text", "image", "vlm")  # canonical concat order, independent of input order


def canonical_sources(sources) -> tuple[str, ...]:
    """Normalize a feature-source selection to the canonical order; reject unknown names."""
    chosen = set(sources)
    unknown = chosen - set(SOURCE_ORDER)
    if unknown or not chosen:
        raise ValueError(f"bad feature sources {sources!r}; pick from {SOURCE_ORDER}")
    return tuple(s for s in SOURCE_ORDER if s in chosen)


def cfg_sources(cfg) -> tuple[str, ...]:
    """The feature-source selection from config (features.sources), defaulting to text+image."""
    feats = cfg.get("features", None)
    return canonical_sources(tuple(feats.sources)) if feats is not None else ("text", "image")


def load_retrieval_data(
    paths: Paths, n_price_buckets: int = 16, cat_min_count: int = 20, sources=("text", "image")
) -> RetrievalData:
    # --- dense multimodal content (aligned to item_idx) ---
    sources = canonical_sources(sources)
    files = {
        "text": paths.text_emb_npy,
        "image": paths.image_emb_npy,
        "vlm": paths.profile_emb_npy,  # VLM item-profile embeddings (Week 8)
    }
    blocks = [np.load(files[s]) for s in sources]
    has = np.load(paths.has_image_npy).astype(np.float32)
    if len({b.shape[0] for b in blocks} | {has.shape[0]}) != 1:
        raise ValueError(f"feature blocks misaligned: {[b.shape for b in blocks]}")
    content = np.concatenate([*blocks, has[:, None]], axis=1).astype(np.float32)
    n_items = content.shape[0]

    # --- structured features via bucket/embedding (item_map order == item_idx) ---
    item_map = pl.read_parquet(paths.item_map_parquet).sort("item_idx")
    meta = pl.read_parquet(paths.meta_parquet)
    m = item_map.join(meta, on="parent_asin", how="left").sort("item_idx")
    assert m.height == n_items, f"meta/content misalignment: {m.height} vs {n_items}"

    prices = np.array([_parse_price(p) for p in m.get_column("price").to_list()], dtype=float)
    price_bucket, price_card = _quantile_buckets(prices, n_price_buckets)
    cat_id, cat_card = _category_leaf_ids(m.get_column("categories").to_list(), cat_min_count)
    item_id = np.arange(n_items, dtype=np.int64)
    cat_features = np.stack([item_id, price_bucket, cat_id], axis=1).astype(np.int64)
    cat_cardinalities = [n_items, price_card, cat_card]
    cat_names = ["item_id", "price_bucket", "category_leaf"]

    # --- interactions / splits ---
    ix = pl.read_parquet(paths.interactions_parquet)
    n_users = int(ix.get_column("user_idx").max()) + 1
    tr = ix.filter(pl.col("split") == "train")
    # polars with_row_index/join yields UInt32; cast to int64 (torch index tensors require it)
    tr_u = tr.get_column("user_idx").to_numpy().astype(np.int64)
    tr_i = tr.get_column("item_idx").to_numpy().astype(np.int64)

    # per-user sum/count of train content (chunked to bound peak memory)
    user_sum = np.zeros((n_users, content.shape[1]), dtype=np.float32)
    user_count = np.bincount(tr_u, minlength=n_users).astype(np.int64)
    chunk = 200_000
    for s in range(0, len(tr_u), chunk):
        np.add.at(user_sum, tr_u[s : s + chunk], content[tr_i[s : s + chunk]])

    train_pairs = np.stack([tr_u, tr_i], axis=1)

    # held-out targets
    valid_item = np.full(n_users, -1, dtype=np.int64)
    test_item = np.full(n_users, -1, dtype=np.int64)
    va = ix.filter(pl.col("split") == "valid")
    te = ix.filter(pl.col("split") == "test")
    valid_item[va.get_column("user_idx").to_numpy()] = va.get_column("item_idx").to_numpy()
    test_item[te.get_column("user_idx").to_numpy()] = te.get_column("item_idx").to_numpy()

    # CSR of seen (train) items per user, for masking at eval
    order = np.argsort(tr_u, kind="stable")
    seen_items = tr_i[order].astype(np.int64)
    seen_indptr = np.concatenate([[0], np.cumsum(user_count)]).astype(np.int64)

    item_pop = np.bincount(tr_i, minlength=n_items).astype(np.int64)

    log.info(
        "retrieval data: users=%s items=%s train_pairs=%s | sources=%s content_dim=%s | "
        "price_buckets=%s category_vocab=%s",
        f"{n_users:,}",
        f"{n_items:,}",
        f"{len(tr_u):,}",
        "+".join(sources),
        content.shape[1],
        price_card,
        cat_card,
    )
    return RetrievalData(
        content=content,
        cat_features=cat_features,
        cat_cardinalities=cat_cardinalities,
        cat_names=cat_names,
        user_sum_content=user_sum,
        user_count=user_count,
        train_pairs=train_pairs,
        valid_item=valid_item,
        test_item=test_item,
        seen_indptr=seen_indptr,
        seen_items=seen_items,
        item_pop=item_pop,
        n_users=n_users,
        n_items=n_items,
    )
