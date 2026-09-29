import unittest

import numpy as np
import pandas as pd

from utils.popularity_evaluator import (
    build_popularity_groups,
    evaluate_popularity_views,
)


class _DatasetStub:
    uid_field = 'userID'
    iid_field = 'itemID'

    def __init__(self, rows):
        self.df = pd.DataFrame(
            rows, columns=(self.uid_field, self.iid_field)
        )


class PopularityEvaluatorTest(unittest.TestCase):
    def test_train_only_popularity_split_is_deterministic(self):
        train = _DatasetStub([
            (0, 0), (1, 0), (2, 0),
            (0, 1), (1, 1),
            (0, 2), (0, 3),
        ])
        popular_mask, counts = build_popularity_groups(
            train, item_count=5, popular_ratio=0.4
        )
        self.assertEqual(
            popular_mask.tolist(), [True, True, False, False, False]
        )
        self.assertEqual(counts.tolist(), [3, 2, 1, 1, 0])

    def test_rank_all_items_then_filter_group_ground_truth(self):
        rankings = {
            'final': np.asarray([[0, 2, 1], [2, 3, 1]])
        }
        users = np.asarray([0, 1])
        ground_truth = {
            0: np.asarray([0, 2]),
            1: np.asarray([1]),
        }
        popular_mask = np.asarray([True, True, False, False])
        results = evaluate_popularity_views(
            rankings,
            users,
            ground_truth,
            popular_mask,
            metrics=('Recall', 'NDCG'),
            topk=(1, 3),
        )['final']

        self.assertEqual(results['popular']['users'], 2)
        self.assertEqual(results['niche']['users'], 1)
        self.assertEqual(results['popular']['metrics']['recall@1'], 0.5)
        self.assertEqual(results['popular']['metrics']['recall@3'], 1.0)
        self.assertEqual(results['niche']['metrics']['recall@1'], 0.0)
        self.assertEqual(results['niche']['metrics']['recall@3'], 1.0)


if __name__ == '__main__':
    unittest.main()
