import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


SRC_DIR = Path(__file__).resolve().parents[1] / 'src'
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import pandas as pd
    from utils.quick_start import quick_start  # noqa: E402
    RUNTIME_IMPORT_ERROR = None
except ModuleNotFoundError as error:
    pd = None
    quick_start = None
    RUNTIME_IMPORT_ERROR = error


class TrainingSmokeTest(unittest.TestCase):
    @staticmethod
    def write_dataset(root):
        dataset_dir = Path(root) / 'baby'
        dataset_dir.mkdir()
        interactions = pd.DataFrame(
            [
                (0, 0, 0), (0, 4, 1), (0, 5, 2),
                (1, 1, 0), (1, 5, 1), (1, 6, 2),
                (2, 2, 0), (2, 6, 1), (2, 7, 2),
                (3, 3, 0), (3, 7, 1), (3, 4, 2),
            ],
            columns=('userID', 'itemID', 'x_label'),
        )
        interactions.to_csv(
            dataset_dir / 'baby.inter', sep='\t', index=False
        )

        rng = np.random.default_rng(999)
        np.save(
            dataset_dir / 'image_feat.npy',
            rng.normal(size=(8, 5)).astype(np.float32),
        )
        np.save(
            dataset_dir / 'text_feat.npy',
            rng.normal(size=(8, 4)).astype(np.float32),
        )

    @unittest.skipIf(
        RUNTIME_IMPORT_ERROR is not None,
        'runtime dependency unavailable: {}'.format(RUNTIME_IMPORT_ERROR),
    )
    def test_one_training_epoch_for_all_models(self):
        for model_name in (
            'PGL', 'PGL_MASKED', 'FREEDOM', 'FREEDOM_MASKED'
        ):
            with self.subTest(model=model_name):
                with tempfile.TemporaryDirectory() as temporary_root:
                    self.write_dataset(temporary_root)
                    quick_start(
                        model=model_name,
                        dataset='baby',
                        config_dict={
                            'data_path': temporary_root,
                            'use_gpu': False,
                            'epochs': 1,
                            'stopping_step': 1,
                            'train_batch_size': 4,
                            'eval_batch_size': 4,
                            'knn_k': 2,
                            'metrics': ['Recall', 'NDCG'],
                            'topk': [1, 2],
                            'valid_metric': 'Recall@1',
                            'hyper_parameters': [],
                            'dropout': 0.2,
                            'reg_weight': 0.0,
                            'mode': 'local',
                            'save_recommended_topk': False,
                            'save_analysis_artifacts': False,
                        },
                        save_model=False,
                    )


if __name__ == '__main__':
    unittest.main()
