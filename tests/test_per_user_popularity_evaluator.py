import os
import sys

import numpy as np


SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from utils.per_user_popularity_evaluator import (  # noqa: E402
    build_niche_delta_rows,
    build_per_user_records,
    recall_ndcg_at_k,
)


def test_empty_ground_truth_returns_null_metrics():
    result = recall_ndcg_at_k([1, 2, 3], [], 3)
    assert result == {'recall@3': None, 'ndcg@3': None}


def test_records_and_niche_delta_are_sorted_by_masked_gain():
    users = np.array([10, 20])
    popular_mask = np.array([True, True, False, False, False, False])
    ground_truth = {
        10: np.array([2]),
        20: np.array([4]),
    }
    rankings = {
        'full': np.array([[0, 1, 3], [4, 0, 1]]),
        'masked': np.array([[2, 0, 1], [0, 1, 3]]),
        'fused': np.array([[2, 1, 0], [4, 1, 0]]),
    }
    records = build_per_user_records(
        users, rankings, ground_truth, popular_mask, k=3
    )
    rows = build_niche_delta_rows(records, k=3)

    assert len(records) == 6
    assert [row['user_id'] for row in rows] == [10, 20]
    assert rows[0]['delta_masked_full_niche_recall@3'] == 1.0
    assert rows[1]['delta_masked_full_niche_recall@3'] == -1.0
    assert rows[0]['rank'] == 1
    assert rows[1]['rank'] == 2


def test_user_without_niche_ground_truth_is_kept_at_bottom():
    users = np.array([1, 2])
    popular_mask = np.array([True, False, False, False])
    ground_truth = {1: np.array([0]), 2: np.array([2])}
    rankings = {
        'full': np.array([[0, 1], [0, 1]]),
        'masked': np.array([[0, 2], [2, 0]]),
        'fused': np.array([[0, 3], [2, 1]]),
    }
    records = build_per_user_records(
        users, rankings, ground_truth, popular_mask, k=2
    )
    rows = build_niche_delta_rows(records, k=2)

    assert [row['user_id'] for row in rows] == [2, 1]
    assert rows[0]['rank'] == 1
    assert rows[1]['rank'] is None
    assert rows[1]['eligible_niche'] is False
