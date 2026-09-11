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

from models.pgl import PGL  # noqa: E402
from models.pgl_masked import PGL_MASKED  # noqa: E402
from models.freedom import FREEDOM  # noqa: E402
from models.freedom_masked import FREEDOM_MASKED  # noqa: E402
from utils.configurator import Config  # noqa: E402
from utils.utils import get_model  # noqa: E402


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


class PGLSmokeTest(unittest.TestCase):
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

    @staticmethod
    def make_config(root):
        return NullableConfig({
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
            'embedding_size': 2,
            'feat_embed_dim': 2,
            'knn_k': 2,
            'lambda_coeff': 0.9,
            'n_mm_layers': 1,
            'n_ui_layers': 1,
            'reg_weight': 0.1,
            'mm_image_weight': 0.1,
            'predefined_epoch': None,
            'dropout': 0.0,
            'mode': 'local',
        })

    def test_original_pgl_forward_loss_and_backward(self):
        with tempfile.TemporaryDirectory() as temporary_root:
            self.write_features(temporary_root)
            model = PGL(self.make_config(temporary_root), FakeTrainData())
            model.pre_epoch_processing()

            users, items = model.forward(model.sub_graph)
            self.assertEqual(tuple(users.shape), (3, 4))
            self.assertEqual(tuple(items.shape), (4, 4))

            interaction = (
                torch.tensor([0, 2]),
                torch.tensor([0, 3]),
                torch.tensor([2, 0]),
            )
            loss = model.calculate_loss(interaction)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            self.assertIsNotNone(model.user_image.weight.grad)

    def test_supported_models_are_importable(self):
        self.assertIs(get_model('PGL'), PGL)
        self.assertIs(get_model('PGL_MASKED'), PGL_MASKED)
        self.assertIs(get_model('FREEDOM'), FREEDOM)
        self.assertIs(get_model('FREEDOM_MASKED'), FREEDOM_MASKED)

    def test_config_loading_does_not_depend_on_working_directory(self):
        original_directory = os.getcwd()
        with tempfile.TemporaryDirectory() as temporary_root:
            try:
                os.chdir(temporary_root)
                config = Config(
                    'PGL_MASKED', 'baby', {'use_gpu': False}
                )
            finally:
                os.chdir(original_directory)

        expected_data_path = Path(__file__).resolve().parents[1] / 'data'
        self.assertEqual(Path(config['data_path']), expected_data_path)
        self.assertEqual(config['model'], 'PGL_MASKED')


if __name__ == '__main__':
    unittest.main()
