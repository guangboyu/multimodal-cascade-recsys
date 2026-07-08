"""Pure-logic tests for ranking metrics (Mann-Whitney AUC with midrank tie handling)."""

from __future__ import annotations

import math

import numpy as np

from vlmrec.ranking.train import _auc


def test_auc_perfect_separation():
    assert _auc(np.array([0.9, 0.8]), np.array([0.2, 0.1])) == 1.0


def test_auc_perfectly_wrong():
    assert _auc(np.array([0.1]), np.array([0.9, 0.8])) == 0.0


def test_auc_all_tied_is_half():
    assert _auc(np.array([0.5, 0.5]), np.array([0.5, 0.5, 0.5])) == 0.5


def test_auc_known_value():
    # pairs: (.8>.6) (.8>.2) (.4<.6) (.4>.2) -> 3 wins of 4
    assert _auc(np.array([0.8, 0.4]), np.array([0.6, 0.2])) == 0.75


def test_auc_tie_counts_half():
    # 4 pairs: (.8>.5)=1, (.8>.5)=1, (.5==.5)=0.5, (.5==.5)=0.5 -> 3/4
    assert _auc(np.array([0.8, 0.5]), np.array([0.5, 0.5])) == 0.75


def test_auc_empty_inputs_are_nan():
    assert math.isnan(_auc(np.array([]), np.array([0.5])))
    assert math.isnan(_auc(np.array([0.5]), np.array([])))
