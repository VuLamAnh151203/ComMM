"""Evaluate predictive and Top-K complementarity from one masked checkpoint."""

import argparse
import json
import os
import pickle
import subprocess

import numpy as np
import torch
import yaml

from utils.complementarity import (
    MaskedModelComplementarityScorer,
    aggregate_seed_results,
    atomic_json_dump,
    id_mapping_hash,
    interaction_hash,
    load_or_create_triplet_cache,
    pairwise_metrics,
    positives_by_user,
    ranking_gained_lost,
    sample_uniform_triplets,
    sha256_file,
    sha256_json,
    upsert_csv,
    validate_triplets,
    write_csv,
)
from utils.configurator import Config
from utils.dataloader import TrainDataLoader
from utils.dataset import RecDataset
from utils.utils import get_model, init_seed


SCRIPT_VERSION = 'pcml-complementarity-v1'
SUPPORTED_MODELS = {'PGL_MASKED', 'FREEDOM_MASKED'}


def _load_checkpoint(path, trusted=False):
    try:
        return torch.load(path, map_location='cpu', weights_only=True)
    except TypeError:
        return torch.load(path, map_location='cpu')
    except pickle.UnpicklingError as error:
        if not trusted:
            raise RuntimeError(
                'Checkpoint safe loading failed. Rerun with '
                '--trusted-checkpoint only for a checkpoint you trust.'
            ) from error
        return torch.load(path, map_location='cpu', weights_only=False)


def _state_and_config(checkpoint):
    if not isinstance(checkpoint, dict):
        raise ValueError('Checkpoint must be a dictionary.')
    if 'model_state_dict' in checkpoint:
        return checkpoint['model_state_dict'], checkpoint.get('config', {})
    if 'state_dict' in checkpoint:
        return checkpoint['state_dict'], checkpoint.get('config', {})
    if checkpoint and all(torch.is_tensor(value) for value in checkpoint.values()):
        return checkpoint, {}
    raise ValueError('Checkpoint does not contain a model state_dict.')


def _load_config_file(path):
    if path is None:
        return {}
    with open(path, 'r', encoding='utf-8') as source:
        if path.lower().endswith('.json'):
            value = json.load(source)
        else:
            value = yaml.safe_load(source)
    if not isinstance(value, dict):
        raise ValueError('--config must contain a mapping/object.')
    return value


def _git_commit(repo_dir):
    try:
        result = subprocess.run(
            ['git', '-C', repo_dir, 'rev-parse', 'HEAD'],
            check=True, capture_output=True, text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _evaluate_pairwise(scorer, triplets, batch_size):
    if batch_size <= 0:
        raise ValueError('pairwise_batch_size must be positive.')
    reference_margins = []
    fused_margins = []
    for start in range(0, len(triplets), batch_size):
        batch = triplets[start:start + batch_size]
        users = batch[:, 0]
        positives = batch[:, 1]
        negatives = batch[:, 2]
        reference_margin = (
            scorer.score_pairs(users, positives, 'reference')
            - scorer.score_pairs(users, negatives, 'reference')
        )
        fused_margin = (
            scorer.score_pairs(users, positives, 'fused')
            - scorer.score_pairs(users, negatives, 'fused')
        )
        reference_margins.append(
            reference_margin.detach().cpu().double().numpy()
        )
        fused_margins.append(fused_margin.detach().cpu().double().numpy())
    return pairwise_metrics(
        np.concatenate(reference_margins), np.concatenate(fused_margins)
    )


def _boundary_tie(scores, candidate_ids, k):
    if candidate_ids.size <= k:
        return False
    ordered = np.sort(scores[candidate_ids])[::-1]
    return bool(ordered[k - 1] == ordered[k])


def _evaluate_full_ranking(
    scorer,
    train_positives,
    test_positives,
    valid_item_ids,
    k,
    user_batch_size,
):
    if k <= 0 or user_batch_size <= 0:
        raise ValueError('k and user_batch_size must be positive.')
    users = np.asarray(sorted(test_positives), dtype=np.int64)
    reference_topk = {}
    fused_topk = {}
    evaluator_fused_topk = {}
    short_candidate_users = 0
    boundary_tie_users = 0
    valid_item_ids = np.asarray(valid_item_ids, dtype=np.int64)

    for start in range(0, users.size, user_batch_size):
        batch_users = users[start:start + user_batch_size]
        reference_scores = scorer.score_all_items(
            batch_users, 'reference'
        ).detach().cpu().numpy()
        fused_scores_tensor = scorer.score_all_items(batch_users, 'fused')
        fused_scores = fused_scores_tensor.detach().cpu().numpy()

        for row_index, user in enumerate(batch_users):
            user = int(user)
            seen = set(int(item) for item in train_positives.get(user, ()))
            candidate_mask = np.ones(valid_item_ids.size, dtype=bool)
            if seen:
                candidate_mask[np.fromiter(seen, dtype=np.int64)] = False
            candidates = valid_item_ids[candidate_mask]
            positives = set(int(item) for item in test_positives[user])
            missing = positives.difference(int(item) for item in candidates)
            if missing:
                raise ValueError(
                    'Test positives are excluded by ranking filters for user '
                    '{}: {}.'.format(user, sorted(missing))
                )
            effective_k = min(k, int(candidates.size))
            if effective_k < k:
                short_candidate_users += 1
            if effective_k == 0:
                raise ValueError('User {} has no ranking candidates.'.format(user))

            reference_order = np.lexsort((
                candidates, -reference_scores[row_index, candidates]
            ))
            fused_order = np.lexsort((
                candidates, -fused_scores[row_index, candidates]
            ))
            reference_topk[user] = candidates[
                reference_order[:effective_k]
            ]
            fused_topk[user] = candidates[fused_order[:effective_k]]
            if _boundary_tie(fused_scores[row_index], candidates, effective_k):
                boundary_tie_users += 1

            # Reproduce the repository evaluator (torch.topk after masking
            # training positives) as an integration cross-check.
            evaluator_row = fused_scores_tensor[row_index].clone()
            if seen:
                seen_tensor = torch.as_tensor(
                    sorted(seen), dtype=torch.long,
                    device=evaluator_row.device,
                )
                evaluator_row[seen_tensor] = -1e10
            evaluator_fused_topk[user] = torch.topk(
                evaluator_row, effective_k
            ).indices.detach().cpu().numpy()

    metrics, per_user = ranking_gained_lost(
        test_positives, reference_topk, fused_topk, k
    )
    evaluator_metrics, _ = ranking_gained_lost(
        test_positives, reference_topk, evaluator_fused_topk, k
    )
    evaluator_recall = evaluator_metrics['fused_recall_at_{}'.format(k)]
    stable_recall = metrics['fused_recall_at_{}'.format(k)]
    difference = stable_recall - evaluator_recall
    if boundary_tie_users == 0 and not np.isclose(
        difference, 0.0, atol=1e-12, rtol=0.0
    ):
        raise AssertionError(
            'Stable fused Recall does not match the repository evaluator.'
        )
    checks = {
        'short_candidate_users': short_candidate_users,
        'topk_boundary_tie_users': boundary_tie_users,
        'repository_evaluator_fused_recall': evaluator_recall,
        'stable_tie_fused_recall': stable_recall,
        'stable_minus_repository_recall': difference,
    }
    return metrics, per_user, checks


def _write_readme(output_dir):
    path = os.path.join(output_dir, 'README.md')
    content = """# PCML complementarity evaluation

This directory contains evaluation-only results. Reference and fused scores
come from the same masked-model checkpoint. Pairwise metrics use the cached
uniform held-out triplets; Top-20 metrics use full-catalog ranking and exclude
training positives exactly like ComMM's test evaluator.

The deterministic ranking tie policy is descending raw score then ascending
item ID. `checks/*.json` reports any boundary ties and comparison with the
repository's `torch.topk` convention. Rates in CSV files are fractions.
Undefined conditional rates are empty/`null`, never silently replaced by zero.

Single checkpoint:

```bash
python evaluate_complementarity.py --checkpoint /path/model.pth \\
  --dataset baby --model PGL_MASKED --device cuda:0 \\
  --output-dir results/complementarity
```

Manifest batch and standalone aggregation:

```bash
python run_complementarity_manifest.py --manifest checkpoints.csv \\
  --device cuda:0 --output-dir results/complementarity
python aggregate_complementarity.py \\
  --output-dir results/complementarity
```
"""
    temporary = path + '.tmp'
    with open(temporary, 'w', encoding='utf-8') as output:
        output.write(content)
    os.replace(temporary, path)


def _update_protocol(path, core, dataset_entry, run_entry):
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as source:
            payload = json.load(source)
        if payload.get('protocol_signature') != core['protocol_signature']:
            raise ValueError(
                'Output directory already contains a different protocol.'
            )
    else:
        payload = dict(core)
        payload['datasets'] = {}
        payload['runs'] = {}
    dataset_key = '{}:{}'.format(
        dataset_entry['dataset'], dataset_entry['id_mapping_hash'][:16]
    )
    payload['datasets'][dataset_key] = dataset_entry
    payload['runs'][run_entry['checkpoint_id']] = run_entry
    atomic_json_dump(payload, path)


def evaluate_checkpoint(args):
    if args.split != 'test':
        raise ValueError('Only the held-out test split is supported.')
    checkpoint_path = os.path.abspath(args.checkpoint)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(checkpoint_path)
    checkpoint = _load_checkpoint(checkpoint_path, args.trusted_checkpoint)
    state_dict, saved_config = _state_and_config(checkpoint)
    overrides = dict(saved_config or {})
    overrides.update(_load_config_file(args.config))

    model_name = args.model or overrides.get('model')
    dataset_name = args.dataset or overrides.get('dataset')
    if not model_name or not dataset_name:
        raise ValueError('Model and dataset must be supplied or saved.')
    model_name = str(model_name).upper()
    dataset_name = str(dataset_name).lower()
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(
            'Complementarity adapter supports only {}.'.format(
                sorted(SUPPORTED_MODELS)
            )
        )
    overrides.pop('device', None)
    if args.data_path:
        overrides['data_path'] = os.path.abspath(args.data_path)
    if args.device:
        device = args.device.lower()
        overrides['use_gpu'] = device != 'cpu'
        if device.startswith('cuda:'):
            overrides['gpu_id'] = int(device.split(':', 1)[1])
    overrides['save_recommended_topk'] = False
    config = Config(
        model=model_name, dataset=dataset_name,
        config_dict=overrides, mg=False,
    )
    training_seed = args.training_seed
    if training_seed is None:
        training_seed = config['seed']
        if isinstance(training_seed, (list, tuple)):
            training_seed = training_seed[0]
    training_seed = int(training_seed)
    config['seed'] = training_seed
    init_seed(training_seed)

    dataset = RecDataset(config)
    train_dataset, validation_dataset, test_dataset = dataset.split()
    # AbstractDataLoader expects inter_num, which RecDataset materializes in
    # its textual statistics method.
    str(train_dataset)
    train_data = TrainDataLoader(
        config, train_dataset,
        batch_size=config['train_batch_size'], shuffle=False,
    )
    model = get_model(model_name)(config, train_data).to(config['device'])
    model.load_state_dict(state_dict, strict=True)
    if hasattr(model, 'post_epoch_processing'):
        model.post_epoch_processing()
    model.eval()
    scorer = MaskedModelComplementarityScorer(model)

    test_positives = positives_by_user(test_dataset)
    train_positives = positives_by_user(train_dataset)
    validation_positives = positives_by_user(validation_dataset)
    if not test_positives:
        raise ValueError('The filtered test split has no users.')
    valid_item_ids = np.arange(dataset.item_num, dtype=np.int64)
    split_hashes = {
        'train': interaction_hash(train_dataset),
        'validation': interaction_hash(validation_dataset),
        'test': interaction_hash(test_dataset),
    }
    mapping_hash = id_mapping_hash(
        dataset.user_num, dataset.item_num, valid_item_ids
    )

    output_dir = os.path.abspath(args.output_dir)
    cache_path = args.triplet_cache or os.path.join(
        output_dir, 'triplets',
        '{}_test_uniform{}_seed{}.npz'.format(
            dataset_name, args.num_negatives, args.sampling_seed
        ),
    )
    expected_cache_metadata = {
        'schema_version': 1,
        'dataset': dataset_name,
        'split': 'test',
        'split_hashes': split_hashes,
        'id_mapping_hash': mapping_hash,
        'valid_catalog_policy': 'all_embedding_rows_0_to_item_num_minus_1',
        'sampler': 'uniform_without_replacement',
        'sampling_seed': int(args.sampling_seed),
        'requested_negatives_per_positive': int(args.num_negatives),
    }
    triplets, cache_metadata = load_or_create_triplet_cache(
        cache_path,
        expected_cache_metadata,
        lambda: sample_uniform_triplets(
            train_positives, validation_positives, test_positives,
            valid_item_ids, args.num_negatives, args.sampling_seed,
        ),
    )
    validate_triplets(
        triplets, train_positives, validation_positives,
        test_positives, valid_item_ids,
    )

    sample_users = np.asarray(sorted(test_positives)[:8], dtype=np.int64)
    fused_score_max_error = scorer.check_against_model(sample_users)
    pairwise = _evaluate_pairwise(
        scorer, triplets, args.pairwise_batch_size
    )
    ranking, per_user, ranking_checks = _evaluate_full_ranking(
        scorer, train_positives, test_positives, valid_item_ids,
        args.k, args.user_batch_size,
    )

    checkpoint_hash = sha256_file(checkpoint_path)
    checkpoint_id = checkpoint_hash[:16]
    config_path = os.path.abspath(args.config) if args.config else ''
    config_hash = (
        sha256_file(config_path) if config_path
        else sha256_json(config.final_config_dict)
    )
    cache_id = cache_metadata['cache_sha256'][:16]
    row = {
        'dataset': dataset_name,
        'model': model_name,
        'training_seed': training_seed,
        'checkpoint_id': checkpoint_id,
        'triplet_cache_id': cache_id,
        'num_triplets': int(len(triplets)),
        'num_pairwise_users': int(np.unique(triplets[:, 0]).size),
        'num_eval_users': ranking['num_eval_users'],
    }
    row.update(pairwise)
    row.update(ranking)
    row.update({
        'skipped_users_empty_negative_pool': cache_metadata[
            'skipped_users_empty_negative_pool'
        ],
        'skipped_positives_empty_negative_pool': cache_metadata[
            'skipped_positives_empty_negative_pool'
        ],
    })

    os.makedirs(os.path.join(output_dir, 'per_user'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'checks'), exist_ok=True)
    per_user_path = os.path.join(
        output_dir, 'per_user',
        '{}_{}_{}_{}.csv'.format(
            dataset_name, model_name, training_seed, checkpoint_id
        ),
    )
    write_csv(per_user_path, per_user)
    checks = {
        'checkpoint_id': checkpoint_id,
        'fused_adapter_max_abs_error': fused_score_max_error,
        'C_hat_loss_identity_error': abs(
            pairwise['C_hat']
            - pairwise['reference_pairwise_loss']
            + pairwise['fused_pairwise_loss']
        ),
        'pairwise_accuracy_identity_error': abs(
            pairwise['fused_pairwise_accuracy']
            - pairwise['reference_pairwise_accuracy']
            - pairwise['delta_pairwise_accuracy']
        ),
        'recall_identity_error': abs(
            ranking['delta_recall_at_{}'.format(args.k)]
            - ranking['gained_at_{}'.format(args.k)]
            + ranking['lost_at_{}'.format(args.k)]
        ),
        'proposition_bound_holds': (
            pairwise['C_hat'] + 1e-10 >= pairwise['B_hat']
        ),
        **ranking_checks,
    }
    checks_path = os.path.join(
        output_dir, 'checks',
        '{}_{}_{}_{}.json'.format(
            dataset_name, model_name, training_seed, checkpoint_id
        ),
    )
    atomic_json_dump(checks, checks_path)

    manifest_row = {
        'dataset': dataset_name,
        'model': model_name,
        'seed': training_seed,
        'checkpoint_path': checkpoint_path,
        'checkpoint_sha256': checkpoint_hash,
        'checkpoint_id': checkpoint_id,
        'config_path': config_path,
        'config_sha256': config_hash,
        'epoch': checkpoint.get('epoch', '') if isinstance(checkpoint, dict) else '',
    }
    upsert_csv(
        os.path.join(output_dir, 'checkpoint_manifest.csv'), manifest_row,
        ('checkpoint_id',),
    )
    upsert_csv(
        os.path.join(output_dir, 'per_seed_metrics.csv'), row,
        ('checkpoint_id', 'triplet_cache_id'),
    )

    source_dir = os.path.dirname(os.path.abspath(__file__))
    repo_dir = os.path.dirname(source_dir)
    protocol_signature = sha256_json({
        'script_version': SCRIPT_VERSION,
        'k': args.k,
        'num_negatives': args.num_negatives,
        'sampling_seed': args.sampling_seed,
        'ranking_filter': 'train_positives_only',
        'tie_policy': 'score_descending_then_item_id_ascending',
        'pairwise_averaging': 'triplet_micro',
        'ranking_averaging': 'user_macro',
    })
    protocol_core = {
        'schema_version': 1,
        'script_version': SCRIPT_VERSION,
        'repository_commit': _git_commit(repo_dir),
        'protocol_signature': protocol_signature,
        'split': 'test',
        'k': args.k,
        'pairwise_sampling': {
            'sampler': 'uniform_without_replacement',
            'sampling_seed': args.sampling_seed,
            'requested_negatives_per_positive': args.num_negatives,
            'negative_exclusions': ['train', 'validation', 'test'],
        },
        'ranking': {
            'scope': 'full_catalog',
            'filter': 'train_positives_only',
            'validation_positives_filtered': False,
            'test_positives_filtered': False,
            'tie_policy': 'score_descending_then_item_id_ascending',
        },
        'averaging': {
            'pairwise': 'triplet_micro',
            'ranking': 'user_macro',
            'seed_std': 'sample_std_ddof_1',
        },
        'tolerances': {
            'identity_atol': 1e-12,
            'score_adapter_atol': 1e-6,
            'score_adapter_rtol': 1e-5,
            'proposition_bound_atol': 1e-10,
        },
    }
    dataset_entry = {
        'dataset': dataset_name,
        'split_hashes': split_hashes,
        'id_mapping_hash': mapping_hash,
        'triplet_cache_path': os.path.abspath(cache_path),
        'triplet_cache_sha256': cache_metadata['cache_sha256'],
        'num_users': dataset.user_num,
        'num_items': dataset.item_num,
    }
    run_entry = {
        'checkpoint_id': checkpoint_id,
        'model': model_name,
        'dataset': dataset_name,
        'training_seed': training_seed,
        'id_mapping_hash': mapping_hash,
        'triplet_cache_id': cache_id,
        'device': str(config['device']),
        'score_dtype': str(scorer.embeddings['fused'][0].dtype),
        'pairwise_batch_size': args.pairwise_batch_size,
        'user_batch_size': args.user_batch_size,
        'reference_score': (
            'dot(full_users, full_items + shared/full mm_items)'
        ),
        'fused_score': 'dot(fused users, final fused items)',
        'reference_is_internal_branch': True,
        'source_functions': {
            'representations': 'models/*_masked.py::_encode',
            'fused_prediction': 'models/*_masked.py::full_sort_predict',
            'adapter': 'utils/complementarity.py::MaskedModelComplementarityScorer',
        },
    }
    _update_protocol(
        os.path.join(output_dir, 'protocol.json'),
        protocol_core, dataset_entry, run_entry,
    )
    _write_readme(output_dir)
    aggregate_seed_results(output_dir)
    print(json.dumps(row, indent=2, ensure_ascii=False))
    return row


def build_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate PCML predictive and Top-K complementarity.'
    )
    parser.add_argument('--dataset')
    parser.add_argument('--model')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config')
    parser.add_argument('--data-path')
    parser.add_argument('--training-seed', type=int)
    parser.add_argument('--split', default='test', choices=('test',))
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--num-negatives', type=int, default=50)
    parser.add_argument('--sampling-seed', type=int, default=20261005)
    parser.add_argument('--triplet-cache')
    parser.add_argument('--pairwise-batch-size', type=int, default=65536)
    parser.add_argument('--user-batch-size', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    parser.add_argument(
        '--output-dir', default='results/complementarity'
    )
    parser.add_argument('--trusted-checkpoint', action='store_true')
    return parser


if __name__ == '__main__':
    evaluate_checkpoint(build_parser().parse_args())
