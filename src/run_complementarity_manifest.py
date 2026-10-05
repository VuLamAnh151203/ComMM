"""Run complementarity evaluation for checkpoints listed in a CSV manifest."""

import argparse
import csv
import os
import subprocess
import sys


def _optional(command, flag, value):
    if value not in (None, ''):
        command.extend((flag, str(value)))


def _manifest_path(value, manifest_dir):
    if value in (None, '') or os.path.isabs(value):
        return value
    return os.path.abspath(os.path.join(manifest_dir, value))


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate every checkpoint in an explicit manifest.'
    )
    parser.add_argument('--manifest', required=True)
    parser.add_argument(
        '--output-dir', default='results/complementarity'
    )
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--num-negatives', type=int, default=50)
    parser.add_argument('--sampling-seed', type=int, default=20261005)
    parser.add_argument('--pairwise-batch-size', type=int, default=65536)
    parser.add_argument('--user-batch-size', type=int, default=256)
    parser.add_argument('--device', default='cpu')
    parser.add_argument(
        '--reference-mode', default='internal-full',
        choices=('internal-full', 'backbone-checkpoint'),
    )
    parser.add_argument('--trusted-checkpoint', action='store_true')
    args = parser.parse_args()

    manifest_path = os.path.abspath(args.manifest)
    with open(manifest_path, 'r', encoding='utf-8', newline='') as source:
        rows = list(csv.DictReader(source))
    required = {'dataset', 'model'}
    if not rows:
        raise ValueError('Input checkpoint manifest is empty.')
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(
            'Input manifest is missing columns: {}.'.format(sorted(missing))
        )
    if not {'checkpoint', 'checkpoint_path'}.intersection(rows[0]):
        raise ValueError(
            "Input manifest needs 'checkpoint' or 'checkpoint_path'."
        )
    if not {'training_seed', 'seed'}.intersection(rows[0]):
        raise ValueError("Input manifest needs 'training_seed' or 'seed'.")

    manifest_dir = os.path.dirname(manifest_path)
    script = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'evaluate_complementarity.py',
    )
    for row in rows:
        checkpoint = row.get('checkpoint') or row.get('checkpoint_path')
        reference_checkpoint = (
            row.get('reference_checkpoint')
            or row.get('reference_checkpoint_path')
        )
        training_seed = row.get('training_seed') or row.get('seed')
        config = row.get('config') or row.get('config_path')
        reference_config = (
            row.get('reference_config')
            or row.get('reference_config_path')
        )
        if not checkpoint or training_seed in (None, ''):
            raise ValueError(
                'Every manifest row needs a checkpoint path and seed.'
            )
        if (
            args.reference_mode == 'backbone-checkpoint'
            and not reference_checkpoint
        ):
            raise ValueError(
                'Every manifest row needs reference_checkpoint when using '
                'the backbone-checkpoint reference mode.'
            )
        command = [
            sys.executable, script,
            '--dataset', row['dataset'],
            '--model', row['model'],
            '--training-seed', training_seed,
            '--checkpoint', _manifest_path(
                checkpoint, manifest_dir
            ),
            '--reference-mode', args.reference_mode,
            '--output-dir', args.output_dir,
            '--k', str(args.k),
            '--num-negatives', str(args.num_negatives),
            '--sampling-seed', str(args.sampling_seed),
            '--pairwise-batch-size', str(args.pairwise_batch_size),
            '--user-batch-size', str(args.user_batch_size),
            '--device', row.get('device') or args.device,
        ]
        _optional(
            command, '--config',
            _manifest_path(config, manifest_dir),
        )
        _optional(
            command, '--reference-checkpoint',
            _manifest_path(reference_checkpoint, manifest_dir),
        )
        _optional(command, '--reference-model', row.get('reference_model'))
        _optional(
            command, '--reference-config',
            _manifest_path(reference_config, manifest_dir),
        )
        _optional(
            command, '--data-path',
            _manifest_path(row.get('data_path'), manifest_dir),
        )
        _optional(
            command, '--triplet-cache',
            _manifest_path(row.get('triplet_cache'), manifest_dir),
        )
        if args.trusted_checkpoint:
            command.append('--trusted-checkpoint')
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
