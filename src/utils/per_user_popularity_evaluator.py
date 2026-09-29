"""Per-user popularity evaluation and masked-vs-full comparison."""

import csv
import json
import os

import numpy as np


GROUPS = ('overall', 'popular', 'niche')


def _as_int_list(values):
    return [int(value) for value in np.asarray(values).tolist()]


def split_user_ground_truth(test_ground_truth, popular_mask):
    """Split each user's test positives into overall/popular/niche sets."""
    popular_mask = np.asarray(popular_mask, dtype=bool)
    grouped = {}
    for user, positives in test_ground_truth.items():
        positives = np.asarray(positives, dtype=np.int64)
        if positives.size and (
            positives.min() < 0 or positives.max() >= popular_mask.size
        ):
            raise ValueError('Ground-truth item ID is outside the catalog.')
        grouped[int(user)] = {
            'overall': positives,
            'popular': positives[popular_mask[positives]],
            'niche': positives[~popular_mask[positives]],
        }
    return grouped


def recall_ndcg_at_k(recommendations, positives, k):
    """Return per-user Recall@K and NDCG@K, or null for empty GT."""
    positives = np.asarray(positives, dtype=np.int64)
    if positives.size == 0:
        return {'recall@{}'.format(k): None, 'ndcg@{}'.format(k): None}

    recommendations = np.asarray(recommendations, dtype=np.int64)[:k]
    hits = np.isin(recommendations, positives, assume_unique=False)
    recall = float(hits.sum() / positives.size)
    discounts = 1.0 / np.log2(np.arange(2, recommendations.size + 2))
    dcg = float((hits * discounts).sum())
    ideal_length = min(k, positives.size)
    ideal_dcg = float(
        (1.0 / np.log2(np.arange(2, ideal_length + 2))).sum()
    )
    ndcg = dcg / ideal_dcg if ideal_dcg else 0.0
    return {
        'recall@{}'.format(k): round(recall, 6),
        'ndcg@{}'.format(k): round(ndcg, 6),
    }


def build_per_user_records(
    users,
    rankings_by_view,
    test_ground_truth,
    popular_mask,
    k=20,
):
    """Build one detailed record for every (user, representation view)."""
    if k <= 0:
        raise ValueError('k must be positive.')
    users = np.asarray(users, dtype=np.int64)
    if np.unique(users).size != users.size:
        raise ValueError('Evaluation users must be unique.')

    grouped_truth = split_user_ground_truth(
        test_ground_truth, popular_mask
    )
    popular_mask = np.asarray(popular_mask, dtype=bool)
    records = []

    for view, rankings in rankings_by_view.items():
        rankings = np.asarray(rankings, dtype=np.int64)
        if rankings.ndim != 2 or rankings.shape[0] != users.size:
            raise ValueError(
                "Rankings for view '{}' do not align with users.".format(
                    view
                )
            )
        if rankings.shape[1] < k:
            raise ValueError(
                "Rankings for view '{}' contain fewer than {} items.".format(
                    view, k
                )
            )

        for row, user in enumerate(users):
            user = int(user)
            recommendations = rankings[row, :k]
            truth = grouped_truth.get(user)
            if truth is None:
                truth = {
                    group: np.empty(0, dtype=np.int64)
                    for group in GROUPS
                }
            metrics = {
                group: recall_ndcg_at_k(
                    recommendations, truth[group], k
                )
                for group in GROUPS
            }
            popular_recommendations = recommendations[
                popular_mask[recommendations]
            ]
            niche_recommendations = recommendations[
                ~popular_mask[recommendations]
            ]
            records.append({
                'user_id': user,
                'view': str(view),
                'recommendations_top{}'.format(k): _as_int_list(
                    recommendations
                ),
                'recommended_popular_items': _as_int_list(
                    popular_recommendations
                ),
                'recommended_niche_items': _as_int_list(
                    niche_recommendations
                ),
                'ground_truth': {
                    group: _as_int_list(truth[group])
                    for group in GROUPS
                },
                'metrics': metrics,
            })
    return records


def build_niche_delta_rows(records, k=20):
    """Rank users by masked-minus-full niche Recall, then niche NDCG."""
    recall_key = 'recall@{}'.format(k)
    ndcg_key = 'ndcg@{}'.format(k)
    recommendation_key = 'recommendations_top{}'.format(k)
    by_user = {}
    for record in records:
        by_user.setdefault(record['user_id'], {})[record['view']] = record

    rows = []
    for user, views in by_user.items():
        if 'full' not in views or 'masked' not in views:
            continue
        full = views['full']
        masked = views['masked']
        fused = views.get('fused')
        full_recall = full['metrics']['niche'][recall_key]
        masked_recall = masked['metrics']['niche'][recall_key]
        full_ndcg = full['metrics']['niche'][ndcg_key]
        masked_ndcg = masked['metrics']['niche'][ndcg_key]
        eligible = full_recall is not None and masked_recall is not None
        rows.append({
            'rank': None,
            'user_id': int(user),
            'eligible_niche': eligible,
            'niche_ground_truth': full['ground_truth']['niche'],
            'full_recommendations_top{}'.format(k): full[
                recommendation_key
            ],
            'masked_recommendations_top{}'.format(k): masked[
                recommendation_key
            ],
            'fused_recommendations_top{}'.format(k): (
                fused[recommendation_key] if fused else []
            ),
            'full_recommended_niche_items': full[
                'recommended_niche_items'
            ],
            'masked_recommended_niche_items': masked[
                'recommended_niche_items'
            ],
            'full_niche_recall@{}'.format(k): full_recall,
            'masked_niche_recall@{}'.format(k): masked_recall,
            'fused_niche_recall@{}'.format(k): (
                fused['metrics']['niche'][recall_key] if fused else None
            ),
            'delta_masked_full_niche_recall@{}'.format(k): (
                round(masked_recall - full_recall, 6)
                if eligible else None
            ),
            'full_niche_ndcg@{}'.format(k): full_ndcg,
            'masked_niche_ndcg@{}'.format(k): masked_ndcg,
            'fused_niche_ndcg@{}'.format(k): (
                fused['metrics']['niche'][ndcg_key] if fused else None
            ),
            'delta_masked_full_niche_ndcg@{}'.format(k): (
                round(masked_ndcg - full_ndcg, 6)
                if eligible else None
            ),
            'full_recommended_niche_count': len(
                full['recommended_niche_items']
            ),
            'masked_recommended_niche_count': len(
                masked['recommended_niche_items']
            ),
            'delta_masked_full_recommended_niche_count': (
                len(masked['recommended_niche_items'])
                - len(full['recommended_niche_items'])
            ),
        })

    recall_delta_key = 'delta_masked_full_niche_recall@{}'.format(k)
    ndcg_delta_key = 'delta_masked_full_niche_ndcg@{}'.format(k)
    rows.sort(key=lambda row: (
        not row['eligible_niche'],
        -row[recall_delta_key] if row['eligible_niche'] else 0.0,
        -row[ndcg_delta_key] if row['eligible_niche'] else 0.0,
        row['user_id'],
    ))
    rank = 0
    for row in rows:
        if row['eligible_niche']:
            rank += 1
            row['rank'] = rank
    return rows


def _json_cell(value):
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'))
    return value


def save_per_user_outputs(records, jsonl_path, csv_path, k=20):
    """Save nested JSONL and a flattened one-row-per-user-view CSV."""
    for path in (jsonl_path, csv_path):
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    with open(jsonl_path, 'w', encoding='utf-8') as output:
        for record in records:
            output.write(json.dumps(record, ensure_ascii=False) + '\n')

    recommendation_key = 'recommendations_top{}'.format(k)
    fieldnames = [
        'user_id', 'view', recommendation_key,
        'recommended_popular_items', 'recommended_niche_items',
        'overall_ground_truth', 'popular_ground_truth',
        'niche_ground_truth',
    ]
    for group in GROUPS:
        fieldnames.extend((
            '{}_recall@{}'.format(group, k),
            '{}_ndcg@{}'.format(group, k),
        ))

    with open(csv_path, 'w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            row = {
                'user_id': record['user_id'],
                'view': record['view'],
                recommendation_key: _json_cell(record[recommendation_key]),
                'recommended_popular_items': _json_cell(
                    record['recommended_popular_items']
                ),
                'recommended_niche_items': _json_cell(
                    record['recommended_niche_items']
                ),
            }
            for group in GROUPS:
                row['{}_ground_truth'.format(group)] = _json_cell(
                    record['ground_truth'][group]
                )
                row['{}_recall@{}'.format(group, k)] = record[
                    'metrics'
                ][group]['recall@{}'.format(k)]
                row['{}_ndcg@{}'.format(group, k)] = record[
                    'metrics'
                ][group]['ndcg@{}'.format(k)]
            writer.writerow(row)


def save_niche_delta_csv(rows, csv_path):
    """Save the sorted masked-minus-full niche comparison."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    if not rows:
        raise ValueError('No users have both full and masked records.')
    fieldnames = list(rows[0])
    with open(csv_path, 'w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for source in rows:
            writer.writerow({
                key: _json_cell(value) for key, value in source.items()
            })
