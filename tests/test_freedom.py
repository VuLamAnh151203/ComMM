import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn


SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.freedom import FREEDOM  # noqa: E402
from models.freedom_masked import FREEDOM_MASKED  # noqa: E402


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
                np.ones(6, dtype=np.float32),
                (
                    np.array([0, 0, 1, 1, 2, 2]),
                    np.array([0, 1, 1, 2, 2, 3]),
                ),
            ),
            shape=(3, 4),
        )

    def inter_matrix(self, form='coo'):
        return self._interactions.asformat(form)


class TestableFREEDOMMasked(FREEDOM_MASKED):
    def _build_or_load_mm_graph(self, config):
        del config
        indices = torch.arange(self.n_items).repeat(2, 1)
        values = torch.ones(self.n_items)
        self.register_buffer(
            'mm_adj',
            torch.sparse_coo_tensor(
                indices, values, (self.n_items, self.n_items)
            ).coalesce(),
        )


class FreedomTestBase(unittest.TestCase):
    @staticmethod
    def write_features(root, image=True, text=True):
        dataset_dir = Path(root) / 'toy'
        dataset_dir.mkdir(exist_ok=True)
        if image:
            np.save(
                dataset_dir / 'image_feat.npy',
                np.array([
                    [1.0, 0.0, 0.0],
                    [0.8, 0.2, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ], dtype=np.float32),
            )
        if text:
            np.save(
                dataset_dir / 'text_feat.npy',
                np.array([
                    [1.0, 0.0],
                    [0.9, 0.1],
                    [0.0, 1.0],
                    [0.2, 0.8],
                ], dtype=np.float32),
            )

    @staticmethod
    def make_config(root, **overrides):
        config = NullableConfig({
            'USER_ID_FIELD': 'user_id',
            'ITEM_ID_FIELD': 'item_id',
            'NEG_PREFIX': 'neg_',
            'train_batch_size': 2,
            'device': torch.device('cpu'),
            'end2end': False,
            'is_multimodal_model': True,
            'data_path': str(root),
            'dataset': 'toy',
            'vision_feature_file': 'image_feat.npy',
            'text_feature_file': 'text_feat.npy',
            'embedding_size': 3,
            'feat_embed_dim': 2,
            'knn_k': 20,
            'n_mm_layers': 1,
            'n_ui_layers': 2,
            'reg_weight': 0.1,
            'mm_image_weight': 0.2,
            'dropout': 0.5,
            'mask_keep_ratio': 0.4,
            'mask_degree_mode': 'full',
            'mask_graph_mode': 'hard',
            'mask_weight': 0.1,
            'mask_binary_weight': 0.1,
            'hard_mask_temperature': 1.0,
            'random_mask_seed': 123,
            'user_embedding_mode': 'separate',
            'item_embedding_mode': 'separate',
            'item_input_mode': 'id',
            'hybrid_mm_weight': 0.5,
            'ui_branch_mode': 'dual',
            'ui_fusion_mode': 'gated_sum',
            'ui_gate_mode': 'separate',
            'mm_gate_mode': 'reuse_ui_item',
            'cl_weight': 0.5,
            'cl_temperature': 0.2,
        })
        config.update(overrides)
        return config

    @staticmethod
    def interaction():
        return (
            torch.tensor([0, 2]),
            torch.tensor([0, 3]),
            torch.tensor([2, 0]),
        )


class FreedomOriginalTest(FreedomTestBase):
    def test_forward_loss_backward_and_cache(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = FREEDOM(self.make_config(root), FakeTrainData())
            self.assertEqual(model.knn_k, 20)
            self.assertEqual(model.mm_adj.shape, (4, 4))
            self.assertTrue(
                (Path(root) / 'toy' / 'mm_adj_freedomdsp_20_2.pt').is_file()
            )

            model.train()
            model.pre_epoch_processing()
            self.assertEqual(
                model.freedom_adj._nnz(), 2 * model.freedom_keep_count
            )
            users, items = model.forward()
            self.assertEqual(tuple(users.shape), (3, 3))
            self.assertEqual(tuple(items.shape), (4, 3))

            loss = model.calculate_loss(self.interaction())
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(model.item_id_embedding.weight.grad)
            self.assertIsNotNone(model.image_embedding.weight.grad)

            model.eval()
            implicit = model.forward()
            explicit = model.forward(model.norm_adj)
            torch.testing.assert_close(implicit[0], explicit[0])
            torch.testing.assert_close(implicit[1], explicit[1])

    def test_cache_shape_is_validated(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            cache = Path(root) / 'toy' / 'mm_adj_freedomdsp_20_2.pt'
            bad = torch.sparse_coo_tensor(
                torch.tensor([[0], [0]]), torch.ones(1), (2, 2)
            )
            torch.save(bad, cache)
            with self.assertRaisesRegex(ValueError, 'Cached multimodal graph'):
                FREEDOM(self.make_config(root), FakeTrainData())


class FreedomMaskedGraphTest(FreedomTestBase):
    def make_model(self, root, **overrides):
        return TestableFREEDOMMasked(
            self.make_config(root, **overrides), FakeTrainData()
        )

    def test_all_mask_modes_and_train_eval_graph_behavior(self):
        for graph_mode in (
            'soft', 'hard', 'double_full', 'svd', 'local_prunning',
            'random_fixed', 'random_dynamic'
        ):
            with self.subTest(graph_mode=graph_mode):
                with tempfile.TemporaryDirectory() as root:
                    self.write_features(root)
                    model = self.make_model(
                        root, mask_graph_mode=graph_mode
                    )
                    model.train()
                    model.pre_epoch_processing()
                    adjacency, probabilities = (
                        model._masked_ui_adjacency()
                    )
                    self.assertEqual(
                        tuple(adjacency.shape), (7, 7)
                    )
                    if graph_mode in {'soft', 'hard'}:
                        self.assertIsNotNone(probabilities)
                    else:
                        self.assertIsNone(probabilities)
                    if graph_mode == 'hard':
                        self.assertEqual(
                            adjacency._nnz(), 2 * model.hard_keep_count
                        )
                    if graph_mode == 'local_prunning':
                        self.assertEqual(
                            adjacency._nnz(), 2 * model.local_keep_count
                        )
                    if graph_mode in {'random_fixed', 'random_dynamic'}:
                        self.assertEqual(
                            adjacency._nnz(), 2 * model.random_keep_count
                        )

                    users, items = model.forward()
                    self.assertEqual(tuple(users.shape), (3, 3))
                    self.assertEqual(tuple(items.shape), (4, 3))
                    loss = model.calculate_loss(self.interaction())
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    if graph_mode in {'soft', 'hard'}:
                        self.assertIsNotNone(model.mask_logits.grad)

                    model.eval()
                    eval_adjacency, _ = model._masked_ui_adjacency()
                    if graph_mode in {
                        'double_full', 'local_prunning'
                    }:
                        torch.testing.assert_close(
                            eval_adjacency.to_dense(),
                            model.norm_adj.to_dense(),
                        )

    def test_fixed_and_dynamic_random_masks(self):
        for graph_mode in ('random_fixed', 'random_dynamic'):
            with self.subTest(graph_mode=graph_mode):
                with tempfile.TemporaryDirectory() as root:
                    self.write_features(root)
                    model = self.make_model(
                        root, mask_graph_mode=graph_mode
                    )
                    self.assertIsNone(model.mask_logits)
                    initial_train_buffer = model.random_train_indices
                    fixed_eval_indices = model.random_eval_indices.clone()

                    model.train()
                    model.pre_epoch_processing()
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

                    restored = self.make_model(
                        root, mask_graph_mode=graph_mode
                    )
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

    def test_freedom_dropout_resampling_does_not_touch_mask_branch(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(root, mask_graph_mode='hard')
            model.train()
            model.pre_epoch_processing()
            hard_indices = model.hard_train_indices.clone()
            masked_before = model._masked_ui_adjacency()[0].to_dense()

            model._resample_freedom_adjacency()

            torch.testing.assert_close(
                model.hard_train_indices, hard_indices
            )
            torch.testing.assert_close(
                model._masked_ui_adjacency()[0].to_dense(), masked_before
            )

    def test_embedding_modes_are_independent(self):
        for user_mode in ('shared', 'separate'):
            for item_mode in ('shared', 'separate'):
                with self.subTest(user=user_mode, item=item_mode):
                    with tempfile.TemporaryDirectory() as root:
                        self.write_features(root)
                        model = self.make_model(
                            root,
                            mask_graph_mode='double_full',
                            user_embedding_mode=user_mode,
                            item_embedding_mode=item_mode,
                            ui_fusion_mode='gated_sum',
                        )
                        self.assertEqual(
                            model.masked_user_embedding is None,
                            user_mode == 'shared',
                        )
                        self.assertEqual(
                            model.masked_item_id_embedding is None,
                            item_mode == 'shared',
                        )
                        loss = model.calculate_loss(self.interaction())
                        loss.backward()
                        if user_mode == 'separate':
                            self.assertIsNotNone(
                                model.masked_user_embedding.weight.grad
                            )
                        if item_mode == 'separate':
                            self.assertIsNotNone(
                                model.masked_item_id_embedding.weight.grad
                            )

    def test_fusion_output_dimensions_preserve_freedom_residual(self):
        cases = (
            ('dual', 'gated_sum', 3),
            ('dual', 'gated_concat', 3),
            ('masked_only', 'gated_sum', 3),
        )
        for branch_mode, fusion_mode, expected_dim in cases:
            with self.subTest(branch=branch_mode, fusion=fusion_mode):
                with tempfile.TemporaryDirectory() as root:
                    self.write_features(root)
                    model = self.make_model(
                        root,
                        mask_graph_mode='double_full',
                        ui_branch_mode=branch_mode,
                        ui_fusion_mode=fusion_mode,
                    )
                    users, items = model.forward()
                    self.assertEqual(users.shape[1], expected_dim)
                    self.assertEqual(items.shape[1], expected_dim)
                    if branch_mode == 'dual' and fusion_mode == 'gated_concat':
                        self.assertIsNotNone(model.user_concat_projection)
                        self.assertIsNotNone(model.item_concat_projection)

    def test_ui_and_multimodal_gate_mode_matrix(self):
        for ui_gate_mode in ('shared', 'separate'):
            for mm_gate_mode in ('reuse_ui_item', 'separate'):
                with self.subTest(ui=ui_gate_mode, mm=mm_gate_mode):
                    with tempfile.TemporaryDirectory() as root:
                        self.write_features(root)
                        model = self.make_model(
                            root,
                            mask_graph_mode='double_full',
                            ui_gate_mode=ui_gate_mode,
                            mm_gate_mode=mm_gate_mode,
                        )
                        representations = model._encode()
                        torch.testing.assert_close(
                            representations['items'],
                            representations['fused_ui_items']
                            + representations['mm_items'],
                        )
                        if ui_gate_mode == 'shared':
                            self.assertIsNotNone(model.fusion_gate)
                            self.assertIsNone(model.user_fusion_gate)
                            self.assertIsNone(model.item_fusion_gate)
                        else:
                            self.assertIsNone(model.fusion_gate)
                            self.assertIsNotNone(model.user_fusion_gate)
                            self.assertIsNotNone(model.item_fusion_gate)
                        if mm_gate_mode == 'reuse_ui_item':
                            self.assertIsNone(model.mm_fusion_gate)
                            torch.testing.assert_close(
                                representations['mm_gate'],
                                representations['item_gate'],
                            )
                        else:
                            self.assertIsNotNone(model.mm_fusion_gate)

                        model.calculate_loss(self.interaction()).backward()
                        if ui_gate_mode == 'shared':
                            self.assertIsNotNone(model.fusion_gate.weight.grad)
                        else:
                            self.assertIsNotNone(
                                model.user_fusion_gate.weight.grad
                            )
                            self.assertIsNotNone(
                                model.item_fusion_gate.weight.grad
                            )
                        if mm_gate_mode == 'separate':
                            self.assertIsNotNone(
                                model.mm_fusion_gate.weight.grad
                            )

    def test_shared_item_table_uses_one_multimodal_representation(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(
                root,
                mask_graph_mode='double_full',
                item_embedding_mode='shared',
                mm_gate_mode='separate',
            )
            representations = model._encode()
            self.assertIsNone(model.masked_item_id_embedding)
            self.assertIsNone(model.mm_fusion_gate)
            self.assertIsNone(representations['mm_gate'])
            torch.testing.assert_close(
                representations['full_mm_items'],
                representations['masked_mm_items'],
            )
            torch.testing.assert_close(
                representations['items'],
                representations['fused_ui_items']
                + representations['mm_items'],
            )

    def test_multimodal_and_hybrid_inputs_feed_both_graphs(self):
        for item_input_mode in ('multimodal', 'hybrid'):
            with self.subTest(item_input_mode=item_input_mode):
                with tempfile.TemporaryDirectory() as root:
                    self.write_features(root)
                    model = self.make_model(
                        root,
                        item_input_mode=item_input_mode,
                        item_embedding_mode='shared',
                        mask_graph_mode='double_full',
                        cl_weight=0.0,
                        reg_weight=0.0,
                    )
                    representations = model._encode()
                    expected_input = model._original_item_table()
                    torch.testing.assert_close(
                        representations['full_mm_items'], expected_input
                    )
                    torch.testing.assert_close(
                        representations['masked_mm_items'], expected_input
                    )
                    self.assertEqual(
                        tuple(representations['items'].shape), (4, 3)
                    )

                    model.calculate_loss(self.interaction()).backward()
                    self.assertIsNotNone(model.image_trs.weight.grad)
                    self.assertIsNotNone(model.text_trs.weight.grad)
                    if item_input_mode == 'hybrid':
                        self.assertIsNotNone(model.hybrid_mm_logit.grad)

    def test_multimodal_concat_keeps_two_modality_dimensions(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(
                root,
                item_input_mode='multimodal_concat',
                item_embedding_mode='shared',
                user_embedding_mode='separate',
                mask_graph_mode='double_full',
                cl_weight=0.0,
                reg_weight=0.0,
            )
            representations = model._encode()
            self.assertEqual(model.branch_embedding_dim, 6)
            self.assertEqual(tuple(representations['users'].shape), (3, 6))
            self.assertEqual(tuple(representations['items'].shape), (4, 6))
            torch.testing.assert_close(
                representations['full_mm_items'],
                representations['masked_mm_items'],
            )

            model.calculate_loss(self.interaction()).backward()
            self.assertIsNotNone(
                model.concat_user_image_embedding.weight.grad
            )
            self.assertIsNotNone(
                model.concat_user_text_embedding.weight.grad
            )
            self.assertIsNotNone(
                model.masked_user_image_embedding.weight.grad
            )
            self.assertIsNotNone(
                model.masked_user_text_embedding.weight.grad
            )
            self.assertIsNotNone(model.image_trs.weight.grad)
            self.assertIsNotNone(model.text_trs.weight.grad)

    def test_contrastive_loss_is_between_pre_fusion_ui_views(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(root, mask_graph_mode='double_full')
            loss = model.calculate_loss(self.interaction())
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(
                float(model.latest_loss_components['contrastive']), 0.0
            )
            loss.backward()
            self.assertIsNotNone(model.user_embedding.weight.grad)
            self.assertIsNotNone(model.masked_user_embedding.weight.grad)

            masked_only = self.make_model(
                root,
                ui_branch_mode='masked_only',
                ui_fusion_mode='gated_sum',
            )
            masked_only.calculate_loss(self.interaction())
            self.assertEqual(
                float(masked_only.latest_loss_components['contrastive']),
                0.0,
            )

    def test_auxiliary_loss_with_both_or_one_modality(self):
        for image, text in ((True, True), (True, False), (False, True)):
            with self.subTest(image=image, text=text):
                with tempfile.TemporaryDirectory() as root:
                    self.write_features(root, image=image, text=text)
                    model = self.make_model(root)
                    loss = model.calculate_loss(self.interaction())
                    self.assertTrue(torch.isfinite(loss))
                    self.assertIn('cl_weight', model.__dict__)
                    self.assertEqual(model.final_embedding_dim, 3)

    def test_state_restore_artifacts_and_deterministic_inference(self):
        with tempfile.TemporaryDirectory() as root:
            self.write_features(root)
            model = self.make_model(root)
            with torch.no_grad():
                model.mask_logits.copy_(
                    torch.linspace(-2.0, 2.0, model.num_interactions)
                )
            model.calculate_loss(self.interaction())
            model.post_epoch_processing()
            model.eval()
            expected = model.forward()

            restored = self.make_model(root)
            restored.load_state_dict(copy.deepcopy(model.state_dict()))
            restored.post_epoch_processing()
            restored.eval()
            actual = restored.forward()
            torch.testing.assert_close(actual[0], expected[0])
            torch.testing.assert_close(actual[1], expected[1])

            artifacts = restored.get_analysis_artifacts()
            self.assertEqual(
                artifacts['metadata']['item_embedding_mode'], 'separate'
            )
            self.assertEqual(
                artifacts['metadata']['final_embedding_dim'], 3
            )
            self.assertEqual(
                artifacts['metadata']['architecture'],
                'freedom_residual_dual_ui',
            )
            self.assertIn(
                'masked_item_id_embedding.weight',
                artifacts['embedding_tables'],
            )
            self.assertEqual(
                int(artifacts['masks']['masked_branch'][
                    'selected_at_keep_ratio'
                ].sum()),
                restored.hard_keep_count,
            )


class FreedomMaskedSparsePropagationTest(unittest.TestCase):
    @staticmethod
    def native(adjacency, initial, layers):
        outputs = [initial]
        current = initial
        for _ in range(layers):
            current = torch.sparse.mm(adjacency, current)
            outputs.append(current)
        return torch.stack(outputs, dim=1).mean(dim=1)

    def test_custom_sparse_backward_matches_native(self):
        indices = torch.tensor(
            [[0, 0, 1, 2, 3], [1, 2, 2, 3, 0]], dtype=torch.long
        )
        initial_data = torch.tensor([
            [0.2, -0.1], [0.5, 0.3], [-0.4, 0.7], [0.8, -0.2]
        ])
        value_data = torch.tensor([0.4, 0.7, 0.2, 0.9, 0.6])

        model = FREEDOM_MASKED.__new__(FREEDOM_MASKED)
        nn.Module.__init__(model)
        model.n_ui_layers = 2

        custom_initial = initial_data.clone().requires_grad_()
        custom_values = value_data.clone().requires_grad_()
        custom_adj = torch.sparse_coo_tensor(
            indices, custom_values, (4, 4)
        ).coalesce()
        actual = model._propagate_ui_graph(custom_adj, custom_initial)
        actual_grads = torch.autograd.grad(
            actual.square().sum(), (custom_values, custom_initial)
        )

        native_initial = initial_data.clone().requires_grad_()
        native_values = value_data.clone().requires_grad_()
        native_adj = torch.sparse_coo_tensor(
            indices, native_values, (4, 4)
        ).coalesce()
        expected = self.native(native_adj, native_initial, 2)
        expected_grads = torch.autograd.grad(
            expected.square().sum(), (native_values, native_initial)
        )

        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual_grads[0], expected_grads[0])
        torch.testing.assert_close(actual_grads[1], expected_grads[1])


if __name__ == '__main__':
    unittest.main()
