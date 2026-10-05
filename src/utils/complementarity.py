"""Core utilities for paired reference-versus-fused complementarity analysis."""

import csv
import hashlib
import json
import math
import os
from collections import defaultdict

import numpy as np
import torch


SCHEMA_VERSION = 1
PAIRWISE_METRIC_COLUMNS = (
    'num_triplets', 'n_ref_wrong', 'n_ref_correct', 'n_corrected',
    'n_damaged', 'reference_pairwise_loss', 'fused_pairwise_loss',
    'C_hat', 'mean_delta', 'mean_delta_ref_wrong',
    'mean_delta_ref_correct', 'C_ref_wrong', 'C_ref_correct',
    'correction_rate', 'damage_rate', 'reference_pairwise_accuracy',
    'fused_pairwise_accuracy', 'delta_pairwise_accuracy', 'B_hat',
)
RANKING_METRIC_COLUMNS = (
    'num_eval_users', 'reference_recall_at_20', 'fused_recall_at_20',
    'gained_at_20', 'lost_at_20', 'delta_recall_at_20',
    'sum_gained_items', 'sum_lost_items',
)


def atomic_json_dump(payload, path):
    """Atomically write JSON without leaving a partial result file."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as output:
        json.dump(payload, output, indent=2, ensure_ascii=False, sort_keys=True)
    os.replace(temporary, path)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, 'rb') as source:
        while True:
            chunk = source.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value):
    encoded = json.dumps(
        value, sort_keys=True, separators=(',', ':'), ensure_ascii=False,
        default=str,
    ).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def interaction_hash(dataset):
    """Hash an interaction split independently of dataframe row order."""
    pairs = dataset.df[[dataset.uid_field, dataset.iid_field]].to_numpy(
        dtype=np.int64, copy=True
    )
    if pairs.size:
        order = np.lexsort((pairs[:, 1], pairs[:, 0]))
        pairs = np.ascontiguousarray(pairs[order], dtype='<i8')
    digest = hashlib.sha256()
    digest.update(np.asarray(pairs.shape, dtype='<i8').tobytes())
    digest.update(pairs.tobytes())
    return digest.hexdigest()


def id_mapping_hash(user_count, item_count, valid_item_ids):
    valid_item_ids = np.ascontiguousarray(valid_item_ids, dtype='<i8')
    digest = hashlib.sha256()
    digest.update(np.asarray([user_count, item_count], dtype='<i8').tobytes())
    digest.update(valid_item_ids.tobytes())
    return digest.hexdigest()


def positives_by_user(dataset):
    grouped = dataset.df.groupby(dataset.uid_field, sort=False)[
        dataset.iid_field
    ]
    return {
        int(user): np.unique(items.to_numpy(dtype=np.int64))
        for user, items in grouped
    }


def sample_uniform_triplets(
    train_positives,
    validation_positives,
    test_positives,
    valid_item_ids,
    num_negatives,
    sampling_seed,
):
    """Sample deterministic held-out triplets in sorted user/positive order."""
    if num_negatives <= 0:
        raise ValueError('num_negatives must be positive.')
    valid_item_ids = np.asarray(valid_item_ids, dtype=np.int64)
    if valid_item_ids.ndim != 1 or valid_item_ids.size == 0:
        raise ValueError('valid_item_ids must be a non-empty vector.')
    if np.unique(valid_item_ids).size != valid_item_ids.size:
        raise ValueError('valid_item_ids contains duplicates.')

    rng = np.random.default_rng(int(sampling_seed))
    rows = []
    skipped_users = 0
    skipped_positives = 0
    sampled_per_positive = []
    valid_set = set(int(item) for item in valid_item_ids)

    for user in sorted(test_positives):
        user_test = np.unique(test_positives[user]).astype(np.int64)
        if user_test.size == 0:
            continue
        observed = set(int(item) for item in user_test)
        observed.update(int(item) for item in train_positives.get(user, ()))
        observed.update(
            int(item) for item in validation_positives.get(user, ())
        )
        pool = np.asarray(
            sorted(valid_set.difference(observed)), dtype=np.int64
        )
        if pool.size == 0:
            skipped_users += 1
            skipped_positives += int(user_test.size)
            continue

        sample_size = min(int(num_negatives), int(pool.size))
        for positive in sorted(int(item) for item in user_test):
            negatives = rng.choice(pool, size=sample_size, replace=False)
            sampled_per_positive.append(sample_size)
            rows.extend(
                (int(user), positive, int(negative))
                for negative in negatives
            )

    if not rows:
        raise ValueError('No valid held-out triplets could be sampled.')
    triplets = np.asarray(rows, dtype=np.int64)
    return triplets, {
        'skipped_users_empty_negative_pool': skipped_users,
        'skipped_positives_empty_negative_pool': skipped_positives,
        'num_test_positives_sampled': len(sampled_per_positive),
        'min_negatives_per_positive': int(min(sampled_per_positive)),
        'max_negatives_per_positive': int(max(sampled_per_positive)),
    }


def triplet_metadata_path(cache_path):
    return os.path.splitext(cache_path)[0] + '.metadata.json'


def load_or_create_triplet_cache(
    cache_path,
    expected_metadata,
    sampler,
):
    """Reuse only an exactly matching cache, otherwise create it once."""
    cache_path = os.path.abspath(cache_path)
    metadata_path = triplet_metadata_path(cache_path)
    cache_exists = os.path.isfile(cache_path)
    metadata_exists = os.path.isfile(metadata_path)
    if cache_exists != metadata_exists:
        raise RuntimeError(
            'Triplet cache and metadata must either both exist or both be absent.'
        )

    if cache_exists:
        with open(metadata_path, 'r', encoding='utf-8') as source:
            metadata = json.load(source)
        mismatches = {
            key: (metadata.get(key), value)
            for key, value in expected_metadata.items()
            if metadata.get(key) != value
        }
        if mismatches:
            raise ValueError(
                'Triplet cache protocol mismatch: {}.'.format(mismatches)
            )
        actual_hash = sha256_file(cache_path)
        if actual_hash != metadata.get('cache_sha256'):
            raise ValueError('Triplet cache SHA-256 does not match metadata.')
        with np.load(cache_path, allow_pickle=False) as archive:
            triplets = archive['triplets']
        if triplets.ndim != 2 or triplets.shape[1] != 3:
            raise ValueError('Cached triplets must have shape [N, 3].')
        return np.asarray(triplets, dtype=np.int64), metadata

    triplets, sampling_details = sampler()
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    temporary = cache_path + '.tmp'
    with open(temporary, 'wb') as output:
        np.savez_compressed(output, triplets=triplets)
    os.replace(temporary, cache_path)
    metadata = dict(expected_metadata)
    metadata.update(sampling_details)
    metadata.update({
        'shape': list(triplets.shape),
        'num_triplets': int(triplets.shape[0]),
        'columns': ['user_id', 'positive_id', 'negative_id'],
        'cache_sha256': sha256_file(cache_path),
    })
    atomic_json_dump(metadata, metadata_path)
    return triplets, metadata


def validate_triplets(
    triplets,
    train_positives,
    validation_positives,
    test_positives,
    valid_item_ids,
):
    """Reject malformed caches, false negatives, and repeated negatives."""
    triplets = np.asarray(triplets, dtype=np.int64)
    if triplets.ndim != 2 or triplets.shape[1] != 3 or not len(triplets):
        raise ValueError('Triplets must have non-empty shape [N, 3].')
    valid_items = set(int(item) for item in valid_item_ids)
    test_pairs = {
        (int(user), int(item))
        for user, items in test_positives.items()
        for item in items
    }
    observed_pairs = set(test_pairs)
    for positives in (train_positives, validation_positives):
        observed_pairs.update(
            (int(user), int(item))
            for user, items in positives.items()
            for item in items
        )
    for user, positive, negative in triplets:
        user = int(user)
        positive = int(positive)
        negative = int(negative)
        if (user, positive) not in test_pairs:
            raise ValueError('Triplet positive is not a test positive.')
        if negative not in valid_items:
            raise ValueError('Triplet negative is outside the valid catalog.')
        if (user, negative) in observed_pairs:
            raise ValueError('Triplet contains an observed item as negative.')
    order = np.lexsort((triplets[:, 2], triplets[:, 1], triplets[:, 0]))
    ordered = triplets[order]
    if len(ordered) > 1 and np.any(np.all(ordered[1:] == ordered[:-1], axis=1)):
        raise ValueError('Duplicate negative within a (user, positive) pair.')
    return True


def _mean_or_none(values):
    return float(np.mean(values, dtype=np.float64)) if values.size else None


def _stable_sigmoid(values):
    values = np.asarray(values, dtype=np.float64)
    result = np.empty_like(values)
    positive = values >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def pairwise_metrics(reference_margins, fused_margins):
    reference = np.asarray(reference_margins, dtype=np.float64)
    fused = np.asarray(fused_margins, dtype=np.float64)
    if reference.shape != fused.shape or reference.ndim != 1:
        raise ValueError('Pairwise margin vectors must have the same 1-D shape.')
    if reference.size == 0:
        raise ValueError('Pairwise margin vectors cannot be empty.')
    if not np.isfinite(reference).all() or not np.isfinite(fused).all():
        raise ValueError('Pairwise margins contain NaN or Inf.')

    reference_losses = np.logaddexp(0.0, -reference)
    fused_losses = np.logaddexp(0.0, -fused)
    contributions = reference_losses - fused_losses
    delta = fused - reference
    ref_wrong = reference <= 0.0
    ref_correct = ~ref_wrong
    corrected = ref_wrong & (fused > 0.0)
    damaged = ref_correct & (fused <= 0.0)

    total = int(reference.size)
    n_wrong = int(ref_wrong.sum())
    n_correct = int(ref_correct.sum())
    n_corrected = int(corrected.sum())
    n_damaged = int(damaged.sum())
    reference_loss = float(reference_losses.mean(dtype=np.float64))
    fused_loss = float(fused_losses.mean(dtype=np.float64))
    c_hat = float(contributions.mean(dtype=np.float64))
    b_terms = _stable_sigmoid(-reference) * delta - delta ** 2 / 8.0
    result = {
        'num_triplets': total,
        'n_ref_wrong': n_wrong,
        'n_ref_correct': n_correct,
        'n_corrected': n_corrected,
        'n_damaged': n_damaged,
        'reference_pairwise_loss': reference_loss,
        'fused_pairwise_loss': fused_loss,
        'C_hat': c_hat,
        'mean_delta': float(delta.mean(dtype=np.float64)),
        'mean_delta_ref_wrong': _mean_or_none(delta[ref_wrong]),
        'mean_delta_ref_correct': _mean_or_none(delta[ref_correct]),
        'C_ref_wrong': _mean_or_none(contributions[ref_wrong]),
        'C_ref_correct': _mean_or_none(contributions[ref_correct]),
        'correction_rate': n_corrected / n_wrong if n_wrong else None,
        'damage_rate': n_damaged / n_correct if n_correct else None,
        'reference_pairwise_accuracy': n_correct / total,
        'fused_pairwise_accuracy': (
            n_correct + n_corrected - n_damaged
        ) / total,
        'delta_pairwise_accuracy': (n_corrected - n_damaged) / total,
        'B_hat': float(b_terms.mean(dtype=np.float64)),
    }
    if not math.isclose(c_hat, reference_loss - fused_loss, abs_tol=1e-12):
        raise AssertionError('C_hat loss-difference identity failed.')
    if result['C_hat'] + 1e-10 < result['B_hat']:
        raise AssertionError('Empirical Proposition 1 bound failed.')
    return result


def ranking_gained_lost(test_positives, reference_topk, fused_topk, k):
    """Compute user-macro Recall/Gained/Lost and per-user diagnostics."""
    rows = []
    for user in sorted(test_positives):
        positives = set(int(item) for item in test_positives[user])
        if not positives:
            continue
        if user not in reference_topk or user not in fused_topk:
            raise KeyError('Missing ranking for test user {}.'.format(user))
        reference = set(int(item) for item in reference_topk[user][:k])
        fused = set(int(item) for item in fused_topk[user][:k])
        reference_hits = positives.intersection(reference)
        fused_hits = positives.intersection(fused)
        gained = fused_hits.difference(reference_hits)
        lost = reference_hits.difference(fused_hits)
        denominator = len(positives)
        rows.append({
            'user_id': int(user),
            'num_test_positives': denominator,
            'reference_hits': len(reference_hits),
            'fused_hits': len(fused_hits),
            'gained_items': len(gained),
            'lost_items': len(lost),
            'reference_recall': len(reference_hits) / denominator,
            'fused_recall': len(fused_hits) / denominator,
            'gained_normalized': len(gained) / denominator,
            'lost_normalized': len(lost) / denominator,
        })
    if not rows:
        raise ValueError('No test users with positives were evaluated.')

    def mean(field):
        return float(np.mean([row[field] for row in rows], dtype=np.float64))

    reference_recall = mean('reference_recall')
    fused_recall = mean('fused_recall')
    gained = mean('gained_normalized')
    lost = mean('lost_normalized')
    delta = fused_recall - reference_recall
    if not math.isclose(delta, gained - lost, abs_tol=1e-12):
        raise AssertionError('Recall gained/lost identity failed.')
    metrics = {
        'num_eval_users': len(rows),
        'reference_recall_at_{}'.format(k): reference_recall,
        'fused_recall_at_{}'.format(k): fused_recall,
        'gained_at_{}'.format(k): gained,
        'lost_at_{}'.format(k): lost,
        'delta_recall_at_{}'.format(k): delta,
        'sum_gained_items': sum(row['gained_items'] for row in rows),
        'sum_lost_items': sum(row['lost_items'] for row in rows),
    }
    return metrics, rows


class MaskedModelComplementarityScorer:
    """Cache exact reference and fused inference representations once."""

    def __init__(self, model):
        self.model = model
        self.model.eval()
        with torch.inference_mode():
            representations = model._encode()
        if representations.get('full_users') is None:
            raise ValueError('Reference/full branch is unavailable.')
        full_mm_items = representations.get('full_mm_items')
        if full_mm_items is None:
            full_mm_items = representations.get('mm_items')
        if full_mm_items is None:
            raise ValueError('Shared item-item representation is unavailable.')
        self.embeddings = {
            'reference': (
                representations['full_users'],
                representations['full_items'] + full_mm_items,
            ),
            'fused': (
                representations['users'], representations['items']
            ),
        }

    def _ids(self, values):
        if torch.is_tensor(values):
            return values.to(
                device=self.embeddings['fused'][0].device,
                dtype=torch.long,
            )
        return torch.as_tensor(
            values,
            dtype=torch.long,
            device=self.embeddings['fused'][0].device,
        )

    @torch.inference_mode()
    def score_pairs(self, users, items, branch):
        if branch not in self.embeddings:
            raise ValueError("branch must be 'reference' or 'fused'.")
        users = self._ids(users)
        items = self._ids(items)
        if users.shape != items.shape:
            raise ValueError('users and items must have matching shapes.')
        user_embeddings, item_embeddings = self.embeddings[branch]
        scores = (
            user_embeddings.index_select(0, users)
            * item_embeddings.index_select(0, items)
        ).sum(dim=-1)
        if not torch.isfinite(scores).all():
            raise ValueError('Raw pair scores contain NaN or Inf.')
        return scores

    @torch.inference_mode()
    def score_all_items(self, users, branch):
        if branch not in self.embeddings:
            raise ValueError("branch must be 'reference' or 'fused'.")
        users = self._ids(users)
        user_embeddings, item_embeddings = self.embeddings[branch]
        scores = torch.matmul(
            user_embeddings.index_select(0, users),
            item_embeddings.transpose(0, 1),
        )
        if not torch.isfinite(scores).all():
            raise ValueError('Raw full-ranking scores contain NaN or Inf.')
        return scores

    @torch.inference_mode()
    def check_against_model(self, users, atol=1e-6, rtol=1e-5):
        users = self._ids(users)
        adapter = self.score_all_items(users, 'fused')
        direct = self.model.full_sort_predict([users, None])
        direct = direct.view_as(adapter)
        torch.testing.assert_close(adapter, direct, atol=atol, rtol=rtol)
        return float((adapter - direct).abs().max().item())


def write_csv(path, rows, fieldnames=None):
    rows = list(rows)
    if not rows and fieldnames is None:
        raise ValueError('fieldnames are required when rows are empty.')
    if fieldnames is None:
        fieldnames = list(rows[0])
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def upsert_csv(path, row, key_fields):
    rows = []
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8', newline='') as source:
            rows = list(csv.DictReader(source))
    key = tuple(str(row[field]) for field in key_fields)
    rows = [
        existing for existing in rows
        if tuple(str(existing.get(field, '')) for field in key_fields) != key
    ]
    rows.append(row)
    write_csv(path, rows, fieldnames=list(row))


def _parse_optional_float(value):
    if value in (None, '', 'None', 'null', 'NA'):
        return None
    return float(value)


def aggregate_seed_results(output_dir):
    """Aggregate per-checkpoint metrics with sample std across training seeds."""
    input_path = os.path.join(output_dir, 'per_seed_metrics.csv')
    if not os.path.isfile(input_path):
        raise FileNotFoundError(input_path)
    with open(input_path, 'r', encoding='utf-8', newline='') as source:
        rows = list(csv.DictReader(source))
    if not rows:
        raise ValueError('per_seed_metrics.csv is empty.')

    dataset_model_caches = defaultdict(set)
    groups = defaultdict(list)
    for row in rows:
        dataset_model_caches[(row['dataset'], row['model'])].add(
            row['triplet_cache_id']
        )
        groups[(row['dataset'], row['model'], row['triplet_cache_id'])].append(row)
    incompatible = {
        key: sorted(cache_ids)
        for key, cache_ids in dataset_model_caches.items()
        if len(cache_ids) > 1
    }
    if incompatible:
        raise ValueError(
            'Cannot aggregate different triplet protocols together: {}.'
            .format(incompatible)
        )
    metric_names = [
        name for name in (*PAIRWISE_METRIC_COLUMNS, *RANKING_METRIC_COLUMNS)
        if name not in {
            'num_triplets', 'n_ref_wrong', 'n_ref_correct', 'n_corrected',
            'n_damaged', 'num_eval_users', 'sum_gained_items',
            'sum_lost_items',
        }
    ]
    summary_rows = []
    for (dataset, model, cache_id), group_rows in sorted(groups.items()):
        seeds = [row['training_seed'] for row in group_rows]
        if len(seeds) != len(set(seeds)):
            raise ValueError(
                'More than one checkpoint is registered for the same '
                'dataset/model/training seed and protocol.'
            )
        summary = {
            'dataset': dataset,
            'model': model,
            'triplet_cache_id': cache_id,
            'n_seeds': len(seeds),
        }
        for metric in metric_names:
            values = [
                value for value in (
                    _parse_optional_float(row.get(metric))
                    for row in group_rows
                ) if value is not None
            ]
            summary[metric + '_mean'] = (
                float(np.mean(values, dtype=np.float64)) if values else None
            )
            summary[metric + '_std'] = (
                float(np.std(values, ddof=1, dtype=np.float64))
                if len(values) > 1 else None
            )
            summary[metric + '_valid_seeds'] = len(values)
        summary_rows.append(summary)

    write_csv(
        os.path.join(output_dir, 'summary_metrics.csv'), summary_rows
    )
    markdown = [
        '| Dataset | Model | Seeds | C_hat | Correction (%) | Damage (%) '
        '| Gained@20 | Lost@20 | Delta R@20 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|',
    ]

    def display(row, metric, scale=1.0):
        mean = row.get(metric + '_mean')
        std = row.get(metric + '_std')
        if mean is None:
            return 'NA'
        if std is None:
            return '{:.6f}'.format(mean * scale)
        return '{:.6f} ± {:.6f}'.format(mean * scale, std * scale)

    for row in summary_rows:
        markdown.append(
            '| {dataset} | {model} | {n_seeds} | {c_hat} | {correction} '
            '| {damage} | {gained} | {lost} | {delta} |'.format(
                dataset=row['dataset'], model=row['model'],
                n_seeds=row['n_seeds'], c_hat=display(row, 'C_hat'),
                correction=display(row, 'correction_rate', 100.0),
                damage=display(row, 'damage_rate', 100.0),
                gained=display(row, 'gained_at_20'),
                lost=display(row, 'lost_at_20'),
                delta=display(row, 'delta_recall_at_20'),
            )
        )
    summary_path = os.path.join(output_dir, 'summary_table.md')
    with open(summary_path + '.tmp', 'w', encoding='utf-8') as output:
        output.write('\n'.join(markdown) + '\n')
    os.replace(summary_path + '.tmp', summary_path)
    return summary_rows
