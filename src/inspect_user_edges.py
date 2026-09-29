"""Inspect learned/effective U-I edge weights for one user checkpoint."""

import argparse
import csv
import json
import os
import pickle
from logging import getLogger

import numpy as np
import torch

from utils.configurator import Config
from utils.dataloader import TrainDataLoader
from utils.dataset import RecDataset
from utils.logger import init_logger
from utils.popularity_evaluator import build_popularity_groups
from utils.utils import get_model, init_seed


SUPPORTED_MODELS = {'PGL_MASKED', 'FREEDOM_MASKED'}


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


def _sparse_pair_values(adjacency, rows, columns):
    """Look up sparse matrix values without constructing a dense matrix."""
    if adjacency.layout == torch.strided:
        row_tensor = torch.as_tensor(
            rows, dtype=torch.long, device=adjacency.device
        )
        column_tensor = torch.as_tensor(
            columns, dtype=torch.long, device=adjacency.device
        )
        values = adjacency[row_tensor, column_tensor]
        return values.detach().cpu().numpy().astype(np.float64, copy=False)
    if adjacency.layout != torch.sparse_coo:
        adjacency = adjacency.to_sparse_coo()

    adjacency = adjacency.coalesce()
    indices = adjacency.indices().detach().cpu().numpy()
    values = adjacency.values().detach().cpu().numpy()
    node_count = int(adjacency.size(1))
    adjacency_keys = (
        indices[0].astype(np.int64) * node_count
        + indices[1].astype(np.int64)
    )
    order = np.argsort(adjacency_keys)
    sorted_keys = adjacency_keys[order]
    target_keys = (
        np.asarray(rows, dtype=np.int64) * node_count
        + np.asarray(columns, dtype=np.int64)
    )
    positions = np.searchsorted(sorted_keys, target_keys)
    output = np.zeros(target_keys.size, dtype=np.float64)
    in_bounds = positions < sorted_keys.size
    matched = np.zeros(target_keys.size, dtype=bool)
    matched[in_bounds] = (
        sorted_keys[positions[in_bounds]] == target_keys[in_bounds]
    )
    output[matched] = values[order[positions[matched]]]
    return output


def _selection_vectors(model):
    """Return diagnostic top-k and actual eval-selection vectors."""
    interaction_count = int(model.num_interactions)
    mode = model.mask_graph_mode
    topk_selected = None
    eval_selected = None

    if model.mask_logits is not None:
        selected_indices = model._select_hard_eval_indices(model.mask_logits)
        topk_selected = torch.zeros(
            interaction_count, dtype=torch.bool,
            device=selected_indices.device,
        )
        topk_selected[selected_indices] = True
        if mode == 'hard':
            eval_selected = topk_selected.clone()
        elif mode == 'soft':
            eval_selected = torch.ones_like(topk_selected)
    elif mode in {'random_fixed', 'random_dynamic'}:
        selected_indices = model.random_eval_indices
        eval_selected = torch.zeros(
            interaction_count, dtype=torch.bool,
            device=selected_indices.device,
        )
        eval_selected[selected_indices] = True
    elif mode in {'double_full', 'local_prunning'}:
        # local_prunning intentionally uses the full graph at evaluation.
        eval_selected = torch.ones(
            interaction_count, dtype=torch.bool,
            device=model.ui_edge_index.device,
        )

    if topk_selected is not None:
        topk_selected = topk_selected.detach().cpu().numpy()
    if eval_selected is not None:
        eval_selected = eval_selected.detach().cpu().numpy()
    return topk_selected, eval_selected


@torch.no_grad()
def _extract_user_edges(
    model,
    user_id,
    item_degrees,
    popular_mask,
    requested_items=None,
):
    if not 0 <= user_id < model.n_users:
        raise ValueError(
            'user_id {} is outside [0, {}).'.format(user_id, model.n_users)
        )

    interaction_count = int(model.num_interactions)
    item_degrees = np.asarray(item_degrees, dtype=np.int64)
    popular_mask = np.asarray(popular_mask, dtype=bool)
    if item_degrees.size != model.n_items:
        raise ValueError('Item degrees do not align with the catalog.')
    if popular_mask.size != model.n_items:
        raise ValueError('Popularity mask does not align with the catalog.')
    forward_edges = model.ui_edge_index[:, :interaction_count]
    edge_users = forward_edges[0].detach().cpu().numpy().astype(np.int64)
    edge_items = (
        forward_edges[1] - model.n_users
    ).detach().cpu().numpy().astype(np.int64)
    edge_ids = np.flatnonzero(edge_users == user_id)

    requested_item_set = None
    if requested_items is not None:
        requested_item_set = set(int(item) for item in requested_items)
        edge_ids = np.asarray(
            [
                edge_id for edge_id in edge_ids
                if int(edge_items[edge_id]) in requested_item_set
            ],
            dtype=np.int64,
        )
    if edge_ids.size == 0:
        suffix = (
            ' after applying --item-ids' if requested_item_set else ''
        )
        raise ValueError(
            'User {} has no training U-I edges{}.'.format(user_id, suffix)
        )

    model.eval()
    model.post_epoch_processing()
    masked_adjacency, _ = model._masked_ui_adjacency()
    target_items = edge_items[edge_ids]
    target_columns = target_items + model.n_users
    effective_values = _sparse_pair_values(
        masked_adjacency,
        np.full(edge_ids.size, user_id, dtype=np.int64),
        target_columns,
    )

    logits = None
    probabilities = None
    if model.mask_logits is not None:
        logits = model.mask_logits.detach().cpu().numpy()
        probabilities = torch.sigmoid(
            model.mask_logits
        ).detach().cpu().numpy()
    topk_selected, eval_selected = _selection_vectors(model)
    full_weights = model.full_norm_edge_weights[
        :interaction_count
    ].detach().cpu().numpy()

    pair_counts = {}
    for edge_user, edge_item in zip(edge_users, edge_items):
        key = (int(edge_user), int(edge_item))
        pair_counts[key] = pair_counts.get(key, 0) + 1

    rows = []
    for local_index, edge_id in enumerate(edge_ids):
        item_id = int(edge_items[edge_id])
        probability = (
            float(probabilities[edge_id])
            if probabilities is not None else None
        )
        rows.append({
            'rank': None,
            'user_id': int(user_id),
            'item_id': item_id,
            'item_train_degree': int(item_degrees[item_id]),
            'item_popularity_group': (
                'popular' if popular_mask[item_id] else 'niche'
            ),
            'is_popular': bool(popular_mask[item_id]),
            'edge_id': int(edge_id),
            'duplicate_pair_count': pair_counts[(user_id, item_id)],
            'mask_graph_mode': model.mask_graph_mode,
            'mask_degree_mode': model.mask_degree_mode,
            'mask_logit': (
                float(logits[edge_id]) if logits is not None else None
            ),
            'mask_probability': probability,
            'topk_selected_at_keep_ratio': (
                bool(topk_selected[edge_id])
                if topk_selected is not None else None
            ),
            'eval_selected': (
                bool(eval_selected[edge_id])
                if eval_selected is not None else None
            ),
            'full_normalized_edge_weight': float(full_weights[edge_id]),
            # Sparse adjacency coalesces duplicate (user, item) entries, so
            # this is explicitly a pair-level propagation coefficient.
            'effective_eval_pair_weight': float(
                effective_values[local_index]
            ),
            'active_in_eval_graph': bool(
                effective_values[local_index] != 0.0
            ),
        })

    rows.sort(key=lambda row: (
        -row['mask_probability']
        if row['mask_probability'] is not None else 0.0,
        -int(row['eval_selected'])
        if row['eval_selected'] is not None else 0,
        -row['effective_eval_pair_weight'],
        row['item_id'],
        row['edge_id'],
    ))
    for rank, row in enumerate(rows, start=1):
        row['rank'] = rank
    return rows


def _save_outputs(rows, metadata, csv_path, json_path):
    for path in (csv_path, json_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(csv_path, 'w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(json_path, 'w', encoding='utf-8') as output:
        json.dump(
            {'metadata': metadata, 'edges': rows},
            output,
            indent=2,
            ensure_ascii=False,
        )


def inspect_checkpoint(args):
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
    if model_name.upper() not in SUPPORTED_MODELS:
        raise ValueError(
            'Edge-mask inspection supports only {}.'.format(
                sorted(SUPPORTED_MODELS)
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
    train_dataset, _, _ = dataset.split()
    logger.info('%s', dataset)
    logger.info('\n====Training====\n%s', train_dataset)
    train_data = TrainDataLoader(
        config,
        train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=False,
    )
    model = get_model(model_name)(config, train_data).to(config['device'])
    model.load_state_dict(state_dict, strict=True)
    popular_mask, item_degrees = build_popularity_groups(
        train_dataset, dataset.item_num, args.popular_ratio
    )
    rows = _extract_user_edges(
        model,
        args.user_id,
        item_degrees,
        popular_mask,
        args.item_ids,
    )

    output_base = '{}.user_{}_edges'.format(
        os.path.splitext(checkpoint_path)[0], args.user_id
    )
    csv_path = os.path.abspath(args.output_csv or output_base + '.csv')
    json_path = os.path.abspath(args.output_json or output_base + '.json')
    returned_item_ids = {row['item_id'] for row in rows}
    missing_requested_items = (
        sorted(set(args.item_ids) - returned_item_ids)
        if args.item_ids else []
    )
    metadata = {
        'repository': 'ComMM',
        'checkpoint': checkpoint_path,
        'checkpoint_epoch': (
            checkpoint.get('epoch') if isinstance(checkpoint, dict) else None
        ),
        'model': model_name,
        'dataset': dataset_name,
        'user_id': args.user_id,
        'requested_item_ids': args.item_ids,
        'requested_items_without_training_edge': missing_requested_items,
        'returned_edges': len(rows),
        'popular_ratio': args.popular_ratio,
        'popular_items': int(popular_mask.sum()),
        'niche_items': int((~popular_mask).sum()),
        'item_degree_definition': (
            'number of item interactions in the training split'
        ),
        'popularity_definition': (
            'top popular_ratio of the complete catalog by train degree; '
            'ties are broken by ascending item_id'
        ),
        'sort': (
            'mask_probability descending, eval selection descending, '
            'effective weight descending, item_id ascending'
        ),
        'edge_scope': 'training U-I interactions only',
        'weight_note': (
            'effective_eval_pair_weight is the coalesced propagation '
            'coefficient for the (user, item) pair in the eval masked graph'
        ),
    }
    _save_outputs(rows, metadata, csv_path, json_path)
    logger.info('Saved user-edge CSV: %s', csv_path)
    logger.info('Saved user-edge JSON: %s', json_path)
    return {'metadata': metadata, 'edges': rows}


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Inspect masked U-I edges for one ComMM user.'
    )
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--user-id', required=True, type=int)
    parser.add_argument('--item-ids', nargs='+', type=int)
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
    parser.add_argument('--output-csv')
    parser.add_argument('--output-json')
    return parser.parse_args()


if __name__ == '__main__':
    inspect_checkpoint(_parse_args())
