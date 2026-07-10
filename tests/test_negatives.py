"""Pure-logic tests for hygienic negative sampling (the cascade-fix core)."""

from __future__ import annotations

import numpy as np

from vlmrec.ranking.data import sample_negatives

N_ITEMS = 12


def _setup():
    # user 0: valid=3 test=4, both inside their candidate row; user 1: valid=7 test=8
    valid_item = np.array([3, 7], dtype=np.int64)
    test_item = np.array([4, 8], dtype=np.int64)
    cand_topk = np.array(
        [[3, 4, 5, 6, 9], [7, 8, 1, 2, 10]],  # held-out positives sit in the candidate pool
        dtype=np.int64,
    )
    return valid_item, test_item, cand_topk


def test_heldout_positives_never_sampled_as_hard_negatives():
    valid_item, test_item, cand_topk = _setup()
    pu = np.array([0, 1] * 64, dtype=np.int64)
    pi = np.array([0, 5] * 64, dtype=np.int64)
    rng = np.random.default_rng(0)
    for _ in range(50):
        negs = sample_negatives(
            rng, pu, pi, N_ITEMS, 4, valid_item, test_item, cand_topk=cand_topk, hard_frac=0.5
        )
        banned = np.stack([pi, valid_item[pu], test_item[pu]], axis=1)
        assert not (negs[:, :, None] == banned[:, None, :]).any()


def test_random_mode_excludes_positive_and_heldout():
    valid_item, test_item, _ = _setup()
    pu = np.zeros(256, dtype=np.int64)
    pi = np.full(256, 2, dtype=np.int64)
    rng = np.random.default_rng(1)
    negs = sample_negatives(rng, pu, pi, N_ITEMS, 4, valid_item, test_item)
    assert negs.shape == (256, 4)
    assert not np.isin(negs, [2, 3, 4]).any()  # pi=2, valid=3, test=4


def test_hard_negatives_drawn_from_candidate_rows():
    valid_item, test_item, cand_topk = _setup()
    pu = np.array([0] * 100 + [1] * 100, dtype=np.int64)
    pi = np.zeros(200, dtype=np.int64)
    rng = np.random.default_rng(2)
    # toy rows are 40% banned, so give the resampler headroom (real rows: <=2 banned of 200);
    # the +1 fallback that may leave the row is covered by the saturated-row test below
    negs = sample_negatives(
        rng,
        pu,
        pi,
        N_ITEMS,
        4,
        valid_item,
        test_item,
        cand_topk=cand_topk,
        hard_frac=1.0,
        max_redraws=64,
    )
    for u in (0, 1):
        allowed = set(cand_topk[u]) - {3, 4, 7, 8, 0}
        got = set(negs[pu == u].reshape(-1).tolist())
        assert got <= allowed


def test_deterministic_under_fixed_seed():
    valid_item, test_item, cand_topk = _setup()
    pu = np.array([0, 1, 0, 1], dtype=np.int64)
    pi = np.array([0, 5, 1, 6], dtype=np.int64)
    a = sample_negatives(
        np.random.default_rng(7), pu, pi, N_ITEMS, 4, valid_item, test_item, cand_topk, 0.5
    )
    b = sample_negatives(
        np.random.default_rng(7), pu, pi, N_ITEMS, 4, valid_item, test_item, cand_topk, 0.5
    )
    np.testing.assert_array_equal(a, b)


def test_saturated_candidate_row_falls_back_without_banned():
    # every candidate of user 0 is banned -> the +1 fallback must still return clean negatives
    valid_item = np.array([1], dtype=np.int64)
    test_item = np.array([2], dtype=np.int64)
    cand_topk = np.array([[0, 1, 2, 0, 1]], dtype=np.int64)  # only banned values
    pu = np.zeros(16, dtype=np.int64)
    pi = np.zeros(16, dtype=np.int64)  # pi=0 -> banned = {0, 1, 2}
    rng = np.random.default_rng(3)
    negs = sample_negatives(
        rng,
        pu,
        pi,
        n_items=6,
        n_neg=4,
        valid_item=valid_item,
        test_item=test_item,
        cand_topk=cand_topk,
        hard_frac=1.0,
    )
    assert not np.isin(negs, [0, 1, 2]).any()
    assert np.isin(negs, [3, 4, 5]).all()


def test_users_without_heldout_items_are_fine():
    valid_item = np.array([-1], dtype=np.int64)  # -1 == no held-out item
    test_item = np.array([-1], dtype=np.int64)
    pu = np.zeros(64, dtype=np.int64)
    pi = np.full(64, 5, dtype=np.int64)
    negs = sample_negatives(np.random.default_rng(4), pu, pi, N_ITEMS, 4, valid_item, test_item)
    assert negs.shape == (64, 4)
    assert not (negs == 5).any()
