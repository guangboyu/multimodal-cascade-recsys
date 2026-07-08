"""Turn raw reviews into a clean interaction table with temporal splits.

Steps (all in polars, operates on the ~millions-row review table):
  1. clean + cast, drop rows missing user/item/timestamp
  2. dedup (user, item) keeping the earliest interaction
  3. iterative **k-core** filtering on BOTH users and items (standard RecSys densification)
  4. **temporal leave-last-out** split: per user, last interaction -> test, 2nd-last -> valid
  5. contiguous id remapping (user_id/parent_asin -> user_idx/item_idx)

The k-core and split steps are factored into pure functions so they can be unit-tested
without any network or model dependency (see tests/test_data_logic.py).
"""

from __future__ import annotations

import json

import polars as pl
from omegaconf import DictConfig

from ..paths import Paths
from ..utils import get_logger, timer

log = get_logger("vlmrec.interactions")


def iterative_k_core(df: pl.DataFrame, k: int, max_iters: int = 20) -> pl.DataFrame:
    """Repeatedly drop users and items with < k interactions until the set is stable."""
    for it in range(max_iters):
        before = df.height
        keep_u = df.group_by("user_id").len().filter(pl.col("len") >= k).select("user_id")
        df = df.join(keep_u, on="user_id", how="semi")
        keep_i = df.group_by("parent_asin").len().filter(pl.col("len") >= k).select("parent_asin")
        df = df.join(keep_i, on="parent_asin", how="semi")
        after = df.height
        log.info("  k-core iter %d: %s -> %s rows", it + 1, f"{before:,}", f"{after:,}")
        if after == before:
            break
    return df


def temporal_leave_last_out(df: pl.DataFrame, min_seq_len: int = 3) -> pl.DataFrame:
    """Add `pos`, `seq_len`, `split` columns via a per-user temporal ordering.

    Requires columns: user_id, timestamp, parent_asin. Users with < min_seq_len kept
    interactions are dropped (cannot form train/valid/test).
    """
    df = df.sort(["user_id", "timestamp", "parent_asin"])
    df = df.with_columns(
        pl.col("user_id").cum_count().over("user_id").alias("pos"),  # 1..n in temporal order
        pl.len().over("user_id").alias("seq_len"),
    )
    df = df.filter(pl.col("seq_len") >= min_seq_len)
    return df.with_columns(
        pl.when(pl.col("pos") == pl.col("seq_len"))
        .then(pl.lit("test"))
        .when(pl.col("pos") == pl.col("seq_len") - 1)
        .then(pl.lit("valid"))
        .otherwise(pl.lit("train"))
        .alias("split")
    )


def _compute_stats(out: pl.DataFrame, n_users: int, n_items: int) -> dict:
    sc = out.group_by("split").len().sort("split")
    split_counts = dict(
        zip(sc.get_column("split").to_list(), sc.get_column("len").to_list(), strict=False)
    )
    n_x = out.height
    denom = n_users * n_items
    return {
        "n_users": n_users,
        "n_items": n_items,
        "n_interactions": n_x,
        "density_pct": round(100 * n_x / denom, 5) if denom else None,
        "avg_interactions_per_user": round(n_x / n_users, 2) if n_users else None,
        "avg_interactions_per_item": round(n_x / n_items, 2) if n_items else None,
        "split_counts": split_counts,
        "strong_positive_frac": round(out.get_column("strong").mean(), 4),
    }


def run(cfg: DictConfig, paths: Paths) -> dict:
    paths.ensure()
    with timer(log, "build interactions"):
        df = pl.read_parquet(paths.reviews_parquet)
        df = df.with_columns(
            pl.col("rating").cast(pl.Float32, strict=False),
            pl.col("timestamp").cast(pl.Int64, strict=False),
        ).drop_nulls(["user_id", "parent_asin", "timestamp"])
        log.info("raw interactions: %s", f"{df.height:,}")

        if bool(cfg.filtering.dedup):
            df = df.sort("timestamp").unique(
                subset=["user_id", "parent_asin"], keep="first", maintain_order=True
            )
            log.info("after dedup (user,item): %s", f"{df.height:,}")

        df = iterative_k_core(df, int(cfg.filtering.k_core), int(cfg.filtering.max_iters))
        if df.height == 0:
            raise RuntimeError(
                "No interactions left after k-core. "
                "Lower filtering.k_core or raise dataset.max_reviews."
            )

        df = temporal_leave_last_out(df, int(cfg.split.min_seq_len))

        # contiguous id remap on the FINAL row set so maps match interactions exactly
        user_map = df.select("user_id").unique(maintain_order=True).with_row_index("user_idx")
        item_map = df.select("parent_asin").unique(maintain_order=True).with_row_index("item_idx")
        df = df.join(user_map, on="user_id").join(item_map, on="parent_asin")

        df = df.with_columns(
            (pl.col("rating") >= float(cfg.dataset.positive_rating)).cast(pl.Int8).alias("strong")
        )
        out = df.select(["user_idx", "item_idx", "rating", "timestamp", "split", "strong"]).sort(
            ["user_idx", "timestamp"]
        )

        out.write_parquet(paths.interactions_parquet)
        user_map.write_parquet(paths.user_map_parquet)
        item_map.write_parquet(paths.item_map_parquet)

        stats = _compute_stats(out, user_map.height, item_map.height)
        paths.stats_json.write_text(json.dumps(stats, indent=2))

    log.info("interactions -> %s", paths.interactions_parquet)
    for key, val in stats.items():
        log.info("  %-26s %s", key, val)
    return stats
