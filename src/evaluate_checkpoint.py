"""Load a ComMM checkpoint and evaluate it on the test split only."""

import argparse
import os
from logging import getLogger

import torch

from utils.configurator import Config
from utils.dataloader import EvalDataLoader, TrainDataLoader
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.topk_evaluator import TopKEvaluator
from utils.utils import dict2str, get_model, init_seed


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        # Compatibility with PyTorch versions without ``weights_only``.
        return torch.load(path, map_location='cpu')


@torch.no_grad()
def _evaluate_test(model, test_data, config):
    model.eval()
    topk_batches = []
    max_topk = max(config['topk'])

    for batched_data in test_data:
        scores = model.full_sort_predict(batched_data)
        seen_items = batched_data[1]
        scores[seen_items[0], seen_items[1]] = -1e10
        topk_batches.append(torch.topk(
            scores, max_topk, dim=-1
        ).indices)

    evaluator = TopKEvaluator(config)
    return evaluator.evaluate(
        topk_batches, test_data, is_test=True, idx=0
    )


def evaluate_checkpoint(args):
    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            'Checkpoint does not exist: {}'.format(checkpoint_path)
        )

    checkpoint = _load_checkpoint(checkpoint_path)
    checkpoint_config = checkpoint.get('config')
    if not isinstance(checkpoint_config, dict):
        raise ValueError(
            'Checkpoint does not contain the saved ComMM config.'
        )
    if 'model_state_dict' not in checkpoint:
        raise ValueError(
            'Checkpoint does not contain model_state_dict.'
        )

    config_overrides = dict(checkpoint_config)
    model_name = args.model or config_overrides.get('model')
    dataset_name = args.dataset or config_overrides.get('dataset')
    if not model_name or not dataset_name:
        raise ValueError(
            'Model and dataset must exist in the checkpoint or CLI.'
        )

    # Config recalculates ``device``. Paths may be overridden when evaluating
    # a checkpoint on a machine different from the training machine.
    config_overrides.pop('device', None)
    if args.gpu_id is not None:
        config_overrides['gpu_id'] = args.gpu_id
    config_overrides['use_gpu'] = not args.cpu
    config_overrides['save_recommended_topk'] = args.save_recommendations
    if args.data_path is not None:
        config_overrides['data_path'] = os.path.abspath(args.data_path)
    if args.recommend_topk is not None:
        config_overrides['recommend_topk'] = os.path.abspath(
            args.recommend_topk
        )

    config = Config(
        model=model_name,
        dataset=dataset_name,
        config_dict=config_overrides,
        mg=False,
    )
    seed = config['seed']
    if isinstance(seed, (list, tuple)):
        seed = seed[0]
        config['seed'] = seed
    init_seed(seed)
    init_logger(config)
    logger = getLogger()
    logger.info('Evaluating checkpoint: %s', checkpoint_path)
    logger.info(config)

    dataset = RecDataset(config)
    train_dataset, valid_dataset, test_dataset = dataset.split()
    # RecDataset currently initializes ``inter_num`` while formatting its
    # statistics; DataLoader expects that attribute to exist.
    logger.info(str(dataset))
    logger.info('\n====Training====\n%s', train_dataset)
    logger.info('\n====Validation====\n%s', valid_dataset)
    logger.info('\n====Testing====\n%s', test_dataset)
    train_data = TrainDataLoader(
        config,
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=False,
    )
    test_data = EvalDataLoader(
        config,
        test_dataset,
        additional_dataset=train_dataset,
        batch_size=config['eval_batch_size'],
    )

    model = get_model(model_name)(config, train_data).to(config['device'])
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    model.post_epoch_processing()

    test_result = _evaluate_test(model, test_data, config)
    logger.info(
        'Checkpoint epoch: %s', checkpoint.get('epoch', 'unknown')
    )
    logger.info('Test result:\n%s', dict2str(test_result))
    return test_result


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate a saved ComMM checkpoint on the test split.'
    )
    parser.add_argument(
        '--checkpoint', required=True,
        help='path to the saved .pth checkpoint',
    )
    parser.add_argument(
        '--model', choices=(
            'PGL', 'PGL_MASKED', 'FREEDOM', 'FREEDOM_MASKED'
        ),
        help='override model name stored in the checkpoint',
    )
    parser.add_argument(
        '--dataset', help='override dataset stored in the checkpoint'
    )
    parser.add_argument(
        '--data-path',
        help='override the checkpoint data path for this machine',
    )
    parser.add_argument(
        '--gpu-id', type=int,
        help='override the GPU id stored in the checkpoint',
    )
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument(
        '--save-recommendations', action='store_true',
        help='also save the test top-k recommendation CSV',
    )
    parser.add_argument(
        '--recommend-topk',
        help='output directory used with --save-recommendations',
    )
    return parser.parse_args()


if __name__ == '__main__':
    evaluate_checkpoint(_parse_args())
