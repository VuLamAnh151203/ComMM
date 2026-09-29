"""Evaluate a ComMM checkpoint on overall, popular, and niche test items."""

import argparse
import json
import os
from logging import getLogger

import numpy as np
import torch

from utils.configurator import Config
from utils.dataloader import EvalDataLoader, TrainDataLoader
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.popularity_evaluator import (
    build_popularity_groups,
    evaluate_popularity_views,
    ground_truth_by_user,
    save_popularity_results,
)
from utils.utils import get_model, init_seed


MASKED_MODELS = {'PGL_MASKED', 'FREEDOM_MASKED'}


def _load_checkpoint(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')


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
    views = {
        'fused': (representations['users'], representations['items'])
    }
    if representations['masked_users'] is not None:
        masked_mm = representations.get('masked_mm_items')
        if masked_mm is None:
            masked_mm = representations['mm_items']
        views['masked'] = (
            representations['masked_users'],
            representations['masked_items'] + masked_mm,
        )
    if representations['full_users'] is not None:
        full_mm = representations.get('full_mm_items')
        if full_mm is None:
            full_mm = representations['mm_items']
        views['full'] = (
            representations['full_users'],
            representations['full_items'] + full_mm,
        )
    return views


@torch.no_grad()
def _collect_rankings(model, test_data, model_name, requested_views, max_k):
    model.eval()
    embeddings = None
    if model_name.upper() in MASKED_MODELS:
        embeddings = _complete_masked_views(model)
        default_views = tuple(
            view for view in ('full', 'masked', 'fused')
            if view in embeddings
        )
        views = tuple(
            'fused' if view == 'final' else view
            for view in (requested_views or default_views)
        )
        views = tuple(dict.fromkeys(views))
        unavailable = sorted(set(views) - set(embeddings))
        if unavailable:
            raise ValueError(
                'Unavailable checkpoint views: {}.'.format(unavailable)
            )
    else:
        if requested_views and tuple(requested_views) != ('final',):
            raise ValueError(
                'Only --views final is available for non-masked models.'
            )
        views = ('final',)

    ranking_batches = {view: [] for view in views}
    user_batches = []
    for batched_data in test_data:
        batch_users = batched_data[0]
        seen_items = batched_data[1]
        user_batches.append(batch_users.detach().cpu())
        for view in views:
            if embeddings is None:
                scores = model.full_sort_predict(batched_data)
                scores = scores.view(batch_users.size(0), -1)
            else:
                user_embeddings, item_embeddings = embeddings[view]
                scores = torch.matmul(
                    user_embeddings[batch_users],
                    item_embeddings.transpose(0, 1),
                )
            scores[seen_items[0], seen_items[1]] = -torch.inf
            ranking_batches[view].append(
                torch.topk(scores, max_k, dim=-1).indices.cpu()
            )

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
    checkpoint = _load_checkpoint(checkpoint_path)
    state_dict, saved_config = _checkpoint_state_and_config(checkpoint)
    config_overrides = dict(saved_config or {})
    config_overrides.update(_load_json(args.config_json))

    model_name = args.model or config_overrides.get('model')
    dataset_name = args.dataset or config_overrides.get('dataset')
    if not model_name or not dataset_name:
        raise ValueError(
            '--model and --dataset are required when absent from checkpoint.'
        )
    config_overrides.pop('device', None)
    if args.data_path:
        config_overrides['data_path'] = os.path.abspath(args.data_path)
    if args.gpu_id is not None:
        config_overrides['gpu_id'] = args.gpu_id
    config_overrides['use_gpu'] = not args.cpu
    config_overrides['save_recommended_topk'] = False
    if args.topk:
        config_overrides['topk'] = sorted(set(args.topk))
    if args.metrics:
        config_overrides['metrics'] = args.metrics

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
    # RecDataset materializes inter_num while formatting its statistics;
    # AbstractDataLoader relies on that field.
    logger.info('%s', dataset)
    logger.info('\n====Training====\n%s', train_dataset)
    logger.info('\n====Testing====\n%s', test_dataset)
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

    configured_topk = config['topk']
    if isinstance(configured_topk, int):
        configured_topk = [configured_topk]
    topk = sorted(set(int(k) for k in configured_topk))
    configured_metrics = config['metrics']
    if isinstance(configured_metrics, str):
        configured_metrics = [configured_metrics]
    if max(topk) > dataset.item_num:
        raise ValueError('max(topk) cannot exceed the catalog size.')
    users, rankings = _collect_rankings(
        model, test_data, model_name, args.views, max(topk)
    )
    popular_mask, item_counts = build_popularity_groups(
        train_dataset, dataset.item_num, args.popular_ratio
    )
    views = evaluate_popularity_views(
        rankings,
        users,
        ground_truth_by_user(test_dataset),
        popular_mask,
        configured_metrics,
        topk,
    )

    output_base = os.path.splitext(checkpoint_path)[0] + '.popularity'
    json_path = os.path.abspath(args.output_json or output_base + '.json')
    csv_path = os.path.abspath(args.output_csv or output_base + '.csv')
    payload = {
        'metadata': {
            'repository': 'ComMM',
            'checkpoint': checkpoint_path,
            'checkpoint_epoch': (
                checkpoint.get('epoch') if isinstance(checkpoint, dict)
                else None
            ),
            'model': model_name,
            'dataset': dataset_name,
            'ranking_scope': 'all_catalog_items_except_train_history',
            'popularity_source': 'train_interactions_only',
            'group_user_policy': 'users_with_at_least_one_group_target',
            'popularity_tie_break': 'ascending_item_id',
            'popular_ratio': args.popular_ratio,
            'popular_items': int(popular_mask.sum()),
            'niche_items': int((~popular_mask).sum()),
            'train_interactions': int(item_counts.sum()),
            'topk': topk,
            'metrics': [str(metric).lower() for metric in configured_metrics],
        },
        'views': views,
    }
    save_popularity_results(payload, json_path, csv_path)
    for view, groups in views.items():
        for group, result in groups.items():
            logger.info(
                '%s/%s users=%d interactions=%d metrics=%s',
                view, group, result['users'],
                result['test_interactions'], result['metrics'],
            )
    logger.info('Saved popularity JSON: %s', json_path)
    logger.info('Saved popularity CSV: %s', csv_path)
    return payload


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Evaluate ComMM checkpoint by train-item popularity.'
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--model')
    parser.add_argument('--dataset')
    parser.add_argument('--data-path')
    parser.add_argument('--config-json')
    parser.add_argument('--gpu-id', type=int)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--popular-ratio', type=float, default=0.2)
    parser.add_argument(
        '--views', nargs='+', choices=('full', 'masked', 'fused', 'final')
    )
    parser.add_argument('--topk', nargs='+', type=int)
    parser.add_argument(
        '--metrics', nargs='+',
        choices=('Recall', 'NDCG', 'Precision', 'MAP'),
    )
    parser.add_argument('--output-json')
    parser.add_argument('--output-csv')
    return parser.parse_args()


if __name__ == '__main__':
    evaluate_checkpoint(_parse_args())
