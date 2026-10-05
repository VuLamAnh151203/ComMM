"""Aggregate PCML complementarity metrics across training seeds."""

import argparse
import json
import os

from utils.complementarity import aggregate_seed_results


def main():
    parser = argparse.ArgumentParser(
        description='Aggregate complementarity metrics across checkpoints.'
    )
    parser.add_argument(
        '--output-dir', default='results/complementarity'
    )
    args = parser.parse_args()
    output_dir = os.path.abspath(args.output_dir)
    summary = aggregate_seed_results(output_dir)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
