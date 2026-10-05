import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from utils.complementarity import (  # noqa: E402
    MaskedModelComplementarityScorer,
    load_or_create_triplet_cache,
    pairwise_metrics,
    ranking_gained_lost,
    sample_uniform_triplets,
    triplet_metadata_path,
    validate_triplets,
)


class FakeMaskedModel:
    def __init__(self):
        self.training = True
        self.full_users = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.full_items = torch.tensor([
            [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]
        ])
        self.mm_items = torch.tensor([
            [0.1, 0.2], [0.2, 0.1], [0.0, 0.3]
        ])
        self.fused_users = torch.tensor([[0.8, 0.2], [0.1, 0.9]])
        self.fused_items = torch.tensor([
            [1.2, 0.1], [0.1, 1.1], [0.9, 1.0]
        ])

    def eval(self):
        self.training = False
        return self

    def _encode(self):
        return {
            'full_users': self.full_users,
            'full_items': self.full_items,
            'mm_items': self.mm_items,
            'users': self.fused_users,
            'items': self.fused_items,
        }

    def full_sort_predict(self, interaction):
        users = interaction[0]
        return torch.matmul(
            self.fused_users[users], self.fused_items.transpose(0, 1)
        )


class ComplementarityMetricsTest(unittest.TestCase):
    def test_identical_margins_have_zero_contribution(self):
        margins = np.array([-0.5, 0.0, 0.4, 1.0])
        result = pairwise_metrics(margins, margins)
        self.assertAlmostEqual(result['C_hat'], 0.0)
        self.assertAlmostEqual(result['mean_delta'], 0.0)
        self.assertEqual(result['n_corrected'], 0)
        self.assertEqual(result['n_damaged'], 0)
        self.assertEqual(result['correction_rate'], 0.0)
        self.assertEqual(result['damage_rate'], 0.0)

    def test_one_wrong_margin_is_corrected(self):
        result = pairwise_metrics(np.array([-0.5]), np.array([0.3]))
        self.assertAlmostEqual(result['C_hat'], 0.41972174, places=7)
        self.assertEqual(result['correction_rate'], 1.0)
        self.assertIsNone(result['damage_rate'])
        self.assertEqual(result['reference_pairwise_accuracy'], 0.0)
        self.assertEqual(result['fused_pairwise_accuracy'], 1.0)

    def test_loss_can_improve_without_correction(self):
        result = pairwise_metrics(np.array([-0.5]), np.array([-0.2]))
        self.assertGreater(result['C_hat'], 0.0)
        self.assertEqual(result['correction_rate'], 0.0)
        self.assertEqual(result['fused_pairwise_accuracy'], 0.0)

    def test_symmetric_correction_and_damage_counts(self):
        reference = np.array([-0.5, 0.4, -0.2, 1.0])
        fused = np.array([0.3, -0.1, -0.1, 1.2])
        result = pairwise_metrics(reference, fused)
        self.assertEqual(result['n_ref_wrong'], 2)
        self.assertEqual(result['n_ref_correct'], 2)
        self.assertEqual(result['n_corrected'], 1)
        self.assertEqual(result['n_damaged'], 1)
        self.assertEqual(result['correction_rate'], 0.5)
        self.assertEqual(result['damage_rate'], 0.5)
        self.assertEqual(result['reference_pairwise_accuracy'], 0.5)
        self.assertEqual(result['fused_pairwise_accuracy'], 0.5)
        self.assertEqual(result['delta_pairwise_accuracy'], 0.0)

    def test_topk_uses_user_macro_averaging(self):
        positives = {0: np.array([1, 2, 3]), 1: np.array([5, 6])}
        reference = {0: np.array([1, 4]), 1: np.array([5, 6])}
        fused = {0: np.array([2, 3]), 1: np.array([5, 7])}
        result, rows = ranking_gained_lost(
            positives, reference, fused, k=2
        )
        self.assertEqual(len(rows), 2)
        self.assertAlmostEqual(result['gained_at_2'], 1.0 / 3.0)
        self.assertAlmostEqual(result['lost_at_2'], 5.0 / 12.0)
        self.assertAlmostEqual(result['reference_recall_at_2'], 2.0 / 3.0)
        self.assertAlmostEqual(result['fused_recall_at_2'], 7.0 / 12.0)
        self.assertAlmostEqual(result['delta_recall_at_2'], -1.0 / 12.0)
        self.assertEqual(result['sum_gained_items'], 2)
        self.assertEqual(result['sum_lost_items'], 2)

    def test_identical_topk_has_no_gained_or_lost_items(self):
        positives = {0: np.array([1, 2]), 1: np.array([3])}
        rankings = {0: np.array([1, 4]), 1: np.array([3, 5])}
        result, _ = ranking_gained_lost(
            positives, rankings, rankings, k=2
        )
        self.assertEqual(result['gained_at_2'], 0.0)
        self.assertEqual(result['lost_at_2'], 0.0)
        self.assertEqual(result['delta_recall_at_2'], 0.0)

    def test_sampling_is_deterministic_and_excludes_all_positives(self):
        train = {0: np.array([0, 1]), 1: np.array([1])}
        validation = {0: np.array([2]), 1: np.array([2])}
        test = {0: np.array([3, 4]), 1: np.array([3])}
        catalog = np.arange(10)
        first, _ = sample_uniform_triplets(
            train, validation, test, catalog, 3, 20261005
        )
        second, _ = sample_uniform_triplets(
            train, validation, test, catalog, 3, 20261005
        )
        np.testing.assert_array_equal(first, second)
        for user, positive, negative in first:
            self.assertIn(positive, test[user])
            excluded = set(train[user]) | set(validation[user]) | set(test[user])
            self.assertNotIn(negative, excluded)
        for user in test:
            for positive in test[user]:
                negatives = first[
                    (first[:, 0] == user) & (first[:, 1] == positive), 2
                ]
                self.assertEqual(len(negatives), len(np.unique(negatives)))
        self.assertTrue(validate_triplets(
            first, train, validation, test, catalog
        ))

    def test_cache_reuse_and_protocol_mismatch(self):
        metadata = {
            'dataset': 'toy', 'split': 'test',
            'split_hashes': {'train': 'a', 'validation': 'b', 'test': 'c'},
            'id_mapping_hash': 'mapping',
            'sampler': 'uniform_without_replacement',
            'sampling_seed': 7,
            'requested_negatives_per_positive': 2,
        }
        expected = np.array([[0, 1, 2], [0, 1, 3]], dtype=np.int64)

        def sampler():
            return expected, {
                'skipped_users_empty_negative_pool': 0,
                'skipped_positives_empty_negative_pool': 0,
            }

        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / 'triplets.npz')
            created, saved_metadata = load_or_create_triplet_cache(
                path, metadata, sampler
            )
            np.testing.assert_array_equal(created, expected)
            self.assertTrue(Path(triplet_metadata_path(path)).is_file())
            reused, _ = load_or_create_triplet_cache(
                path, metadata,
                lambda: self.fail('Matching cache should be reused.'),
            )
            np.testing.assert_array_equal(reused, expected)
            with self.assertRaises(ValueError):
                load_or_create_triplet_cache(
                    path, {**metadata, 'sampling_seed': 8}, sampler
                )
            self.assertEqual(saved_metadata['num_triplets'], 2)

    def test_extreme_margins_are_numerically_stable(self):
        result = pairwise_metrics(
            np.array([-1000.0, 1000.0]),
            np.array([-999.0, 999.0]),
        )
        for key in (
            'reference_pairwise_loss', 'fused_pairwise_loss',
            'C_hat', 'B_hat',
        ):
            self.assertTrue(math.isfinite(result[key]))

    def test_adapter_scores_match_direct_model_and_pair_lookup(self):
        model = FakeMaskedModel()
        scorer = MaskedModelComplementarityScorer(model)
        self.assertFalse(model.training)
        self.assertEqual(scorer.check_against_model([0, 1]), 0.0)
        users = torch.tensor([0, 1, 0])
        items = torch.tensor([0, 1, 2])
        pair_scores = scorer.score_pairs(users, items, 'fused')
        full_scores = scorer.score_all_items(users, 'fused')
        torch.testing.assert_close(
            pair_scores, full_scores[torch.arange(3), items]
        )
        expected_reference_items = model.full_items + model.mm_items
        torch.testing.assert_close(
            scorer.embeddings['reference'][1], expected_reference_items
        )


if __name__ == '__main__':
    unittest.main()
