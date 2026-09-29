"""Evaluate ComMM masked-model views at the individual-user level."""

import argparse
import json
import os
import pickle
from logging import getLogger

import torch

from utils.configurator import Config
from utils.dataloader import EvalDataLoader, TrainDataLoader
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.per_user_popularity_evaluator import (
    build_niche_delta_rows,
    build_per_user_records,
    save_niche_delta_csv,
    save_per_user_outputs,
)
from utils.popularity_evaluator import (
    build_popularity_groups,
    ground_truth_by_user,
)
from utils.utils import get_model, init_seed


MASKED_MODELS = {'PGL_MASKED', 'FREEDOM_MASKED'}


def _load_checkpoint(path, trusted_checkpoint=False):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')
    except pickle.UnpicklingError as error:
        if not trusted_checkpoint:
            raise RuntimeError(
                'Safe checkpoint loading rejected Python/NumPy objects. '
                'Use --trusted-checkpoint only for a checkpoint you trust.'
            ) from error
        return torch.load(path, map_location='cpu', weights_only=False)


def _checkpoint_state_and_config(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint must be a dictionary or state_dict.')
    if 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict'], checkpoint.get('config', {})
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict'], checkpoint.get('config', {})
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint, {}
    raise ValueError('Could not find a model state_dict in the checkpoint.')


def _load_json(path):
    if path is None:
        return {}
    with open(path, 'r', encoding='utf-8') as source:
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError('--config-json must contain a JSON object.')
    return value


def _complete_masked_views(model):
    representations = model._encode()
    if representations.get('full_users') is None:
        raise ValueError(
            'This analysis requires ui_branch_mode=dual; the checkpoint '
            'does not expose a full branch.'
        )
    if representations.get('masked_users') is None:
        raise ValueError('The checkpoint does not expose a masked branch.')

    full_mm = representations.get('full_mm_items')
    masked_mm = representations.get('masked_mm_items')
    if full_mm is None:
        full_mm = representations['mm_items']
    if masked_mm is None:
        masked_mm = representations['mm_items']
    return {
        'full': (
            representations['full_users'],
            representations['full_items'] + full_mm,
        ),
        'masked': (
            representations['masked_users'],
            representations['masked_items'] + masked_mm,
        ),
        'fused': (
            representations['users'], representations['items']
        ),
    }


@torch.no_grad()
def _collect_rankings(model, test_data, k):
    model.eval()
    embeddings = _complete_masked_views(model)
    ranking_batches = {view: [] for view in embeddings}
    user_batches = []

    for batched_data in test_data:
        batch_users = batched_data[0]
        seen_items = batched_data[1]
        user_batches.append(batch_users.detach().cpu())
        for view, (user_embeddings, item_embeddings) in embeddings.items():
            scores = torch.matmul(
                user_embeddings[batch_users],
                item_embeddings.transpose(0, 1),
            )
            scores[seen_items[0], seen_items[1]] = -torch.inf
            ranking_batches[view].append(
                torch.topk(scores, k, dim=-1).indices.cpu()
            )

    if not user_batches:
        raise ValueError('The test loader produced no users.')
    users = torch.cat(user_batches, dim=0).numpy()
    rankings = {
        view: torch.cat(batches, dim=0).numpy()
        for view, batches in ranking_batches.items()
    }
    return users, rankings


def evaluate_checkpoint(args):
    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)

    checkpoint = _load_checkpoint(
        checkpoint_path, args.trusted_checkpoint
    )
    state_dict, saved_config = _checkpoint_state_and_config(checkpoint)
    config_overrides = dict(saved_config or {})
    config_overrides.update(_load_json(args.config_json))
    model_name = args.model or config_overrides.get('model')
    dataset_name = args.dataset or config_overrides.get('dataset')
    if not model_name or not dataset_name:
        raise ValueError(
            '--model and --dataset are required when absent from checkpoint.'
        )
    if model_name.upper() not in MASKED_MODELS:
        raise ValueError(
            'Per-branch analysis supports only {}.'.format(
                sorted(MASKED_MODELS)
            )
        )

    config_overrides.pop('device', None)
    if args.data_path:
        config_overrides['data_path'] = os.path.abspath(args.data_path)
    if args.gpu_id is not None:
        config_overrides['gpu_id'] = args.gpu_id
    config_overrides['use_gpu'] = not args.cpu
    config_overrides['save_recommended_topk'] = False

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

    dataset = RecDataset(config)
    train_dataset, _, test_dataset = dataset.split()
    logger.info('%s', dataset)
    logger.info('\n====Training====\n%s', train_dataset)
    logger.info('\n====Testing====\n%s', test_dataset)
    if args.k <= 0:
        raise ValueError('k must be positive.')
    if args.k > dataset.item_num:
        raise ValueError('k cannot exceed the catalog size.')

    train_data = TrainDataLoader(
        config, train_dataset,
        batch_size=config['train_batch_size'], shuffle=False,
    )
    test_data = EvalDataLoader(
        config, test_dataset, additional_dataset=train_dataset,
        batch_size=config['eval_batch_size'],
    )
    model = get_model(model_name)(config, train_data).to(config['device'])
    model.load_state_dict(state_dict, strict=True)
    model.post_epoch_processing()

    users, rankings = _collect_rankings(model, test_data, args.k)
    popular_mask, item_counts = build_popularity_groups(
        train_dataset, dataset.item_num, args.popular_ratio
    )
    records = build_per_user_records(
        users,
        rankings,
        ground_truth_by_user(test_dataset),
        popular_mask,
        k=args.k,
    )
    delta_rows = build_niche_delta_rows(records, k=args.k)

    output_base = os.path.splitext(checkpoint_path)[0]
    jsonl_path = os.path.abspath(
        args.output_jsonl or output_base + '.per_user.jsonl'
    )
    csv_path = os.path.abspath(
        args.output_csv or output_base + '.per_user.csv'
    )
    delta_path = os.path.abspath(
        args.output_delta_csv or output_base + '.niche_delta.csv'
    )
    metadata_path = os.path.abspath(
        args.output_metadata_json
        or output_base + '.per_user.metadata.json'
    )
    save_per_user_outputs(records, jsonl_path, csv_path, k=args.k)
    save_niche_delta_csv(delta_rows, delta_path)

    metadata = {
        'repository': 'ComMM',
        'checkpoint': checkpoint_path,
        'checkpoint_epoch': (
            checkpoint.get('epoch') if isinstance(checkpoint, dict) else None
        ),
        'model': model_name,
        'dataset': dataset_name,
        'id_space': 'internal numeric IDs from the interaction file',
        'views': ['full', 'masked', 'fused'],
        'k': args.k,
        'popular_ratio': args.popular_ratio,
        'popular_items': int(popular_mask.sum()),
        'niche_items': int((~popular_mask).sum()),
        'train_interactions': int(item_counts.sum()),
        'evaluated_users': int(users.size),
        'ranking_scope': 'all catalog items except train history',
        'delta_sort': [
            'masked_niche_recall minus full_niche_recall descending',
            'masked_niche_ndcg minus full_niche_ndcg descending',
            'user_id ascending',
        ],
        'users_without_niche_ground_truth': (
            'retained at the bottom with null metrics and null rank'
        ),
        'outputs': {
            'jsonl': jsonl_path,
            'per_user_csv': csv_path,
            'niche_delta_csv': delta_path,
        },
    }
    os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
    with open(metadata_path, 'w', encoding='utf-8') as output:
        json.dump(metadata, output, indent=2, ensure_ascii=False)

    logger.info('Saved per-user JSONL: %s', jsonl_path)
    logger.info('Saved per-user CSV: %s', csv_path)
    logger.info('Saved niche delta CSV: %s', delta_path)
    logger.info('Saved metadata JSON: %s', metadata_path)
    return metadata


def _parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Evaluate full/masked/fused ComMM representations per user.'
        )
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument(
        '--trusted-checkpoint', action='store_true',
        help='Allow legacy pickled checkpoints from a trusted source.',
    )
    parser.add_argument('--model')
    parser.add_argument('--dataset')
    parser.add_argument('--data-path')
    parser.add_argument('--config-json')
    parser.add_argument('--gpu-id', type=int)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--popular-ratio', type=float, default=0.2)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--output-jsonl')
    parser.add_argument('--output-csv')
    parser.add_argument('--output-delta-csv')
    parser.add_argument('--output-metadata-json')
    return parser.parse_args()


if __name__ == '__main__':
    evaluate_checkpoint(_parse_args())
