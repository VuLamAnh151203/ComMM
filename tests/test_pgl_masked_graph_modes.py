import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch


SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.pgl_masked import PGL_MASKED  # noqa: E402


class NullableConfig(dict):
    def __getitem__(self, key):
        return self.get(key)


class FakeDatasetStats:
    def get_user_num(self):
        return 3

    def get_item_num(self):
        return 4


class FakeTrainData:
    def __init__(self):
        self.dataset = FakeDatasetStats()
        self._interactions = sp.coo_matrix(
            (
                np.ones(5, dtype=np.float32),
                (
                    np.array([0, 0, 1, 2, 2]),
                    np.array([0, 1, 2, 1, 3]),
                ),
            ),
            shape=(3, 4),
        )

    def inter_matrix(self, form='coo'):
        return self._interactions.asformat(form)


class TestablePGLMasked(PGL_MASKED):
    def _build_or_load_mm_graph(self, config):
        indices = torch.arange(self.n_items).repeat(2, 1)
        values = torch.ones(self.n_items)
        adjacency = torch.sparse_coo_tensor(
            indices, values, (self.n_items, self.n_items)
        ).coalesce()
        self.register_buffer('mm_adj', adjacency)


class PGLGraphModeTest(unittest.TestCase):
    def make_config(
        self,
        root,
        graph_mode,
        ui_branch_mode='dual',
        ui_fusion_mode='gated_sum',
        cl_mode='auto',
    ):
        return NullableConfig({
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'NEG_PREFIX': 'neg_',
            'train_batch_size': 2,
            'device': torch.device('cpu'),
            'end2end': False,
            'is_multimodal_model': True,
            'data_path': str(root) + os.sep,
            'dataset': 'toy',
            'vision_feature_file': 'image_feat.npy',
            'text_feature_file': 'text_feat.npy',
            'embedding_size': 2,
            'feat_embed_dim': 2,
            'knn_k': 2,
            'n_mm_layers': 1,
            'n_ui_layers': 1,
            'mask_keep_ratio': 0.4,
            'mask_degree_mode': 'full',
            'mask_graph_mode': graph_mode,
            'random_mask_seed': 123,
            'user_embedding_mode': 'separate',
            'ui_branch_mode': ui_branch_mode,
            'ui_fusion_mode': ui_fusion_mode,
            'cl_weight': 0.05,
            'cl_temperature': 0.2,
            'cl_mode': cl_mode,
            'dropout': 0.0,
            'mask_weight': 0.0,
        })

    @staticmethod
    def write_features(root):
        dataset_dir = Path(root) / 'toy'
        dataset_dir.mkdir()
        np.save(
            dataset_dir / 'image_feat.npy',
            np.array([
                [1.0, 0.0, 0.0],
                [0.8, 0.2, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ], dtype=np.float32),
        )
        np.save(
            dataset_dir / 'text_feat.npy',
            np.array([
                [1.0, 0.0],
                [0.9, 0.1],
                [0.0, 1.0],
                [0.2, 0.8],
            ], dtype=np.float32),
        )

    def make_model(
        self,
        root,
        graph_mode,
        ui_branch_mode='dual',
        ui_fusion_mode='gated_sum',
        cl_mode='auto',
    ):
        return TestablePGLMasked(
            self.make_config(
                root,
                graph_mode,
                ui_branch_mode,
                ui_fusion_mode,
                cl_mode,
            ),
            FakeTrainData(),
        )

    def assert_forward_and_loss(self, model):
        users, items = model.forward()
        self.assertEqual(tuple(users.shape), (3, 4))
        self.assertEqual(tuple(items.shape), (4, 4))

        interaction = (
            torch.tensor([0, 2]),
            torch.tensor([0, 3]),
            torch.tensor([2, 0]),
        )
        loss = model.calculate_loss(interaction)
        self.assertTrue(torch.isfinite(loss))
        return loss

    def test_svd_uses_full_and_static_svd_branches(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            self.write_features(temporary_root)
            model = self.make_model(temporary_root, 'SVD')

            self.assertEqual(model.mask_graph_mode, 'svd')
            self.assertIsNone(model.mask_logits)
            svd_adjacency, mask = model._masked_ui_adjacency()
            self.assertIs(svd_adjacency, model.svd_adj)
            self.assertIsNone(mask)
            self.assertEqual(tuple(svd_adjacency.shape), (7, 7))
            self.assertGreater(svd_adjacency._nnz(), 0)

            restored = self.make_model(temporary_root, 'svd')
            restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(
                restored.svd_adj.to_dense(), model.svd_adj.to_dense()
            )
            self.assert_forward_and_loss(model)

    def test_local_prunning_trains_on_subgraph_and_infers_on_full_graph(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            self.write_features(temporary_root)
            model = self.make_model(temporary_root, 'local_prunning')

            self.assertIsNone(model.mask_logits)
            self.assertEqual(model.local_keep_count, 2)
            self.assertEqual(model.local_pruned_adj._nnz(), 4)
            model.pre_epoch_processing()
            local_adjacency, mask = model._masked_ui_adjacency()
            self.assertIs(local_adjacency, model.local_pruned_adj)
            self.assertIsNone(mask)
            self.assertEqual(local_adjacency._nnz(), 4)

            restored = self.make_model(
                temporary_root, 'local_prunning'
            )
            restored.load_state_dict(model.state_dict())
            torch.testing.assert_close(
                restored.local_pruned_adj.to_dense(),
                model.local_pruned_adj.to_dense(),
            )

            model.eval()
            inference_adjacency, inference_mask = (
                model._masked_ui_adjacency()
            )
            self.assertIs(inference_adjacency, model.norm_adj)
            self.assertIsNone(inference_mask)
            model.train()
            self.assert_forward_and_loss(model)

    def test_gated_concat_projects_back_to_ui_dimension(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            self.write_features(temporary_root)
            model = self.make_model(
                temporary_root,
                'double_full',
                ui_fusion_mode='gated_concat',
            )

            self.assertEqual(model.fusion_projection.in_features, 8)
            self.assertEqual(model.fusion_projection.out_features, 4)
            users, items = model.forward()
            self.assertEqual(tuple(users.shape), (3, 4))
            self.assertEqual(tuple(items.shape), (4, 4))

            interaction = (
                torch.tensor([0, 2]),
                torch.tensor([0, 3]),
                torch.tensor([2, 0]),
            )
            loss = model.calculate_loss(interaction)
            loss.backward()
            self.assertIsNotNone(model.fusion_projection.weight.grad)
            self.assertEqual(
                model.get_analysis_artifacts()['metadata'][
                    'ui_fusion_mode'
                ],
                'gated_concat',
            )

    def test_branch_dropout_contrastive_modes(self):
        interaction = (
            torch.tensor([0, 2]),
            torch.tensor([0, 3]),
            torch.tensor([2, 0]),
        )
        for cl_mode in ('branch', 'dropout', 'branch_and_dropout'):
            with self.subTest(cl_mode=cl_mode):
                with tempfile.TemporaryDirectory() as temporary_root:
                    self.write_features(temporary_root)
                    model = self.make_model(
                        temporary_root,
                        'double_full',
                        cl_mode=cl_mode,
                    )
                    loss = model.calculate_loss(interaction)
                    self.assertTrue(torch.isfinite(loss))

                    components = model.latest_loss_components
                    branch_loss = components['branch_contrastive']
                    dropout_loss = components['dropout_contrastive']
                    if cl_mode == 'branch':
                        self.assertGreater(float(branch_loss), 0.0)
                        self.assertEqual(float(dropout_loss), 0.0)
                    elif cl_mode == 'dropout':
                        self.assertEqual(float(branch_loss), 0.0)
                        self.assertGreater(float(dropout_loss), 0.0)
                    else:
                        self.assertGreater(float(branch_loss), 0.0)
                        self.assertGreater(float(dropout_loss), 0.0)
                        torch.testing.assert_close(
                            components['contrastive'],
                            0.5 * (branch_loss + dropout_loss),
                        )

    def test_fixed_and_dynamic_random_masks(self):
        for graph_mode in ('random_fixed', 'random_dynamic'):
            with self.subTest(graph_mode=graph_mode):
                with tempfile.TemporaryDirectory() as temporary_root:
                    self.write_features(temporary_root)
                    model = self.make_model(temporary_root, graph_mode)
                    self.assertIsNone(model.mask_logits)
                    initial_train_buffer = model.random_train_indices
                    fixed_eval_indices = model.random_eval_indices.clone()

                    model.train()
                    model.pre_epoch_processing()
                    adjacency, probabilities = model._masked_ui_adjacency()
                    self.assertIsNone(probabilities)
                    self.assertEqual(
                        adjacency._nnz(), 2 * model.random_keep_count
                    )
                    self.assert_forward_and_loss(model).backward()
                    if graph_mode == 'random_fixed':
                        self.assertIs(
                            model.random_train_indices,
                            initial_train_buffer,
                        )
                        torch.testing.assert_close(
                            model.random_train_indices,
                            model.random_eval_indices,
                        )
                    else:
                        self.assertIsNot(
                            model.random_train_indices,
                            initial_train_buffer,
                        )
                        torch.testing.assert_close(
                            model.random_eval_indices, fixed_eval_indices
                        )

                    model.eval()
                    expected_eval = model._masked_ui_adjacency()[0].to_dense()
                    model.train()
                    model.pre_epoch_processing()
                    model.eval()
                    torch.testing.assert_close(
                        model._masked_ui_adjacency()[0].to_dense(),
                        expected_eval,
                    )

                    restored = self.make_model(temporary_root, graph_mode)
                    restored.load_state_dict(model.state_dict())
                    restored.eval()
                    torch.testing.assert_close(
                        restored._masked_ui_adjacency()[0].to_dense(),
                        expected_eval,
                    )
                    artifacts = restored.get_analysis_artifacts()
                    self.assertEqual(
                        int(artifacts['masks']['random_branch'][
                            'selected_at_keep_ratio'
                        ].sum()),
                        restored.random_keep_count,
                    )


if __name__ == '__main__':
    unittest.main()
