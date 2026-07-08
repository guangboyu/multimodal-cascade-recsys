"""Pure-logic tests for post-processing: MMR, greedy DPP, score fusion, business rules."""

from __future__ import annotations

import numpy as np

from vlmrec.rerank.postprocess import (
    category_cap,
    dpp_greedy,
    fuse_scores,
    intra_list_diversity,
    mmr,
)


def _unit(v):
    v = np.asarray(v, dtype=np.float64)
    return v / np.linalg.norm(v)


def test_mmr_lambda_one_is_pure_relevance():
    scores = np.array([0.3, 0.9, 0.1, 0.7])
    emb = np.stack([_unit(np.random.default_rng(i).normal(size=8)) for i in range(4)])
    order = mmr(scores, emb, k=4, lam=1.0)
    assert order.tolist() == np.argsort(-scores).tolist()


def test_mmr_penalizes_near_duplicates():
    # items 0 and 1 are identical embeddings; 2 is orthogonal with lower relevance
    e0 = _unit([1.0, 0.0])
    scores = np.array([1.0, 0.99, 0.5])
    emb = np.stack([e0, e0, _unit([0.0, 1.0])])
    order = mmr(scores, emb, k=2, lam=0.5)
    assert order.tolist() == [0, 2]  # the duplicate of item 0 is skipped


def test_mmr_returns_at_most_k_unique_indices():
    rng = np.random.default_rng(0)
    scores = rng.random(10)
    emb = np.stack([_unit(rng.normal(size=4)) for _ in range(10)])
    order = mmr(scores, emb, k=6, lam=0.7)
    assert len(order) == 6 == len(set(order.tolist()))


def test_dpp_on_orthogonal_embeddings_is_topk_by_score():
    scores = np.array([0.2, 0.9, 0.5, 0.7])
    emb = np.eye(4)  # orthogonal items: no diversity penalty applies
    order = dpp_greedy(scores, emb, k=3)
    assert order.tolist() == np.argsort(-scores)[:3].tolist()


def test_dpp_prefers_diverse_item_over_duplicate():
    e0 = _unit([1.0, 0.0])
    scores = np.array([1.0, 0.999, 0.6])
    emb = np.stack([e0, e0, _unit([0.0, 1.0])])
    order = dpp_greedy(scores, emb, k=2)
    assert order.tolist() == [0, 2]


def test_fuse_scores_monotone_in_click_and_sat():
    sat = np.array([0.5, 0.5])
    assert fuse_scores(np.array([0.9, 0.3]), sat)[0] > fuse_scores(np.array([0.9, 0.3]), sat)[1]
    click = np.array([0.5, 0.5])
    fused = fuse_scores(click, np.array([0.9, 0.2]))
    assert fused[0] > fused[1]


def test_category_cap_preserves_order_and_caps():
    order = np.array([3, 1, 0, 2])
    categories = np.array([7, 7, 7, 9])  # items 0,1,2 share category 7; item 3 is category 9
    out = category_cap(order, categories, cap=2)
    assert out.tolist() == [3, 1, 0]  # third category-7 item in order (2) dropped, order kept


def test_intra_list_diversity_bounds():
    same = np.stack([_unit([1.0, 0.0])] * 3)
    assert intra_list_diversity(same) == 0.0  # identical items: no diversity
    ortho = np.eye(3)
    assert intra_list_diversity(ortho) == 1.0  # orthogonal items
    opposite = np.stack([_unit([1.0, 0.0]), _unit([-1.0, 0.0])])
    assert intra_list_diversity(opposite) == 2.0  # antipodal: the maximum
    assert intra_list_diversity(ortho[:1]) == 0.0  # singleton list
