"""Pure-logic tests for the interaction-building core (no network, no GPU)."""

from __future__ import annotations

import polars as pl

from vlmrec.data.build_interactions import iterative_k_core, temporal_leave_last_out


def _interactions(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={"user_id": pl.Utf8, "parent_asin": pl.Utf8, "timestamp": pl.Int64},
        orient="row",
    )


def test_k_core_drops_sparse_users_and_orphan_items():
    # A and B interact with i1,i2,i3 each; C interacts only with i4 (a single interaction).
    rows: list[tuple] = []
    for ts, item in enumerate(["i1", "i2", "i3"]):
        rows.append(("A", item, 10 + ts))
        rows.append(("B", item, 20 + ts))
    rows.append(("C", "i4", 5))
    df = _interactions(rows)

    out = iterative_k_core(df, k=2)

    assert set(out["user_id"].to_list()) == {"A", "B"}  # C (<2 interactions) dropped
    assert "i4" not in set(out["parent_asin"].to_list())  # i4 orphaned once C is gone
    assert out.height == 6


def test_leave_last_out_assigns_temporal_splits():
    rows = [
        ("A", "i1", 10),
        ("A", "i2", 20),
        ("A", "i3", 30),
        ("B", "i1", 5),
        ("B", "i2", 6),  # only 2 interactions -> dropped at min_seq_len=3
    ]
    out = temporal_leave_last_out(_interactions(rows), min_seq_len=3)

    assert set(out["user_id"].to_list()) == {"A"}
    by_ts = {r["timestamp"]: r["split"] for r in out.iter_rows(named=True)}
    assert by_ts == {10: "train", 20: "valid", 30: "test"}
