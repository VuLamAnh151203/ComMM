"""Utilities for train-popularity grouped top-k evaluation."""

import csv
import json
import math
import os

import numpy as np


SUPPORTED_METRICS = ('recall', 'ndcg', 'precision', 'map')


def build_popularity_groups(train_dataset, item_count, popular_ratio):
    """Split the complete catalog using item frequency in the train split."""
    if not 0.0 < popular_ratio < 1.0:
        raise ValueError('popular_ratio must be strictly between 0 and 1.')
    if item_count <= 0:
        raise ValueError('item_count must be positive.')

    item_field = train_dataset.iid_field
    item_ids = train_dataset.df[item_field].to_numpy(dtype=np.int64)
    counts = np.bincount(item_ids, minlength=item_count)
    popular_count = max(1, int(math.ceil(item_count * popular_ratio)))

    # Descending train frequency, then ascending item ID for deterministic
    # behavior when multiple items have the same frequency.
    catalog_ids = np.arange(item_count, dtype=np.int64)
    order = np.lexsort((catalog_ids, -counts))
    popular_items = order[:popular_count]
    popular_mask = np.zeros(item_count, dtype=bool)
    popular_mask[popular_items] = True
    return popular_mask, counts


def ground_truth_by_user(test_dataset):
    """Return unique test item IDs for each user."""
    user_field = test_dataset.uid_field
    item_field = test_dataset.iid_field
    grouped = test_dataset.df.groupby(user_field, sort=False)[item_field]
    return {
        int(user): np.unique(items.to_numpy(dtype=np.int64))
        for user, items in grouped
    }


def _validate_metrics(metrics):
    normalized = tuple(str(metric).lower() for metric in metrics)
    unsupported = sorted(set(normalized) - set(SUPPORTED_METRICS))
    if unsupported:
        raise ValueError(
            'Unsupported popularity metrics: {}.'.format(unsupported)
        )
    return normalized


def _group_metrics(topk_items, users, ground_truth, metrics, topk):
    max_k = max(topk)
    discounts = 1.0 / np.log2(np.arange(2, max_k + 2))
    totals = {
        metric: np.zeros(len(topk), dtype=np.float64)
        for metric in metrics
    }
    eligible_users = 0
    interaction_count = 0

    for row, user in enumerate(users):
        positives = ground_truth.get(int(user))
        if positives is None or positives.size == 0:
            continue
        eligible_users += 1
        interaction_count += int(positives.size)
        hits = np.isin(topk_items[row, :max_k], positives)
        cumulative_hits = np.cumsum(hits)
        precision_at_rank = cumulative_hits / np.arange(1, max_k + 1)
        cumulative_dcg = np.cumsum(hits * discounts)

        for index, k in enumerate(topk):
            hit_count = float(cumulative_hits[k - 1])
            if 'recall' in totals:
                totals['recall'][index] += hit_count / positives.size
            if 'precision' in totals:
                totals['precision'][index] += hit_count / k
            if 'ndcg' in totals:
                ideal_length = min(k, positives.size)
                ideal_dcg = discounts[:ideal_length].sum()
                totals['ndcg'][index] += (
                    cumulative_dcg[k - 1] / ideal_dcg
                )
            if 'map' in totals:
                normalization = min(k, positives.size)
                totals['map'][index] += (
                    (precision_at_rank[:k] * hits[:k]).sum()
                    / normalization
                )

    values = {}
    for metric in metrics:
        for index, k in enumerate(topk):
            value = (
                round(
                    float(totals[metric][index] / eligible_users), 6
                )
                if eligible_users else None
            )
            values['{}@{}'.format(metric, k)] = value
    return {
        'users': eligible_users,
        'test_interactions': interaction_count,
        'metrics': values,
    }


def evaluate_popularity_views(
    topk_by_view,
    users,
    test_ground_truth,
    popular_mask,
    metrics,
    topk,
):
    """Evaluate rankings over all items against overall/popular/niche GT."""
    metrics = _validate_metrics(metrics)
    topk = tuple(sorted(set(int(k) for k in topk)))
    if not topk or topk[0] <= 0:
        raise ValueError('topk must contain positive integers.')

    users = np.asarray(users, dtype=np.int64)
    grouped_truth = {'overall': {}, 'popular': {}, 'niche': {}}
    for user, positives in test_ground_truth.items():
        positives = np.asarray(positives, dtype=np.int64)
        grouped_truth['overall'][user] = positives
        grouped_truth['popular'][user] = positives[popular_mask[positives]]
        grouped_truth['niche'][user] = positives[~popular_mask[positives]]

    results = {}
    for view, rankings in topk_by_view.items():
        rankings = np.asarray(rankings, dtype=np.int64)
        if rankings.ndim != 2 or rankings.shape[0] != users.size:
            raise ValueError(
                "Rankings for view '{}' do not align with users.".format(
                    view
                )
            )
        if rankings.shape[1] < max(topk):
            raise ValueError(
                "Rankings for view '{}' contain fewer than max(topk) "
                'items.'.format(view)
            )
        results[view] = {
            group: _group_metrics(
                rankings, users, truth, metrics, topk
            )
            for group, truth in grouped_truth.items()
        }
    return results


def save_popularity_results(payload, json_path, csv_path):
    """Write a detailed JSON file and a flat CSV table."""
    for path in (json_path, csv_path):
        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)

    with open(json_path, 'w', encoding='utf-8') as output:
        json.dump(payload, output, indent=2, ensure_ascii=False)

    metric_names = []
    for view_results in payload['views'].values():
        for group_result in view_results.values():
            metric_names = list(group_result['metrics'])
            break
        if metric_names:
            break

    fieldnames = [
        'repository', 'model', 'dataset', 'view', 'group',
        'users', 'test_interactions',
    ] + metric_names
    with open(csv_path, 'w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        metadata = payload['metadata']
        for view, view_results in payload['views'].items():
            for group, group_result in view_results.items():
                row = {
                    'repository': metadata['repository'],
                    'model': metadata['model'],
                    'dataset': metadata['dataset'],
                    'view': view,
                    'group': group,
                    'users': group_result['users'],
                    'test_interactions': (
                        group_result['test_interactions']
                    ),
                }
                row.update(group_result['metrics'])
                writer.writerow(row)
