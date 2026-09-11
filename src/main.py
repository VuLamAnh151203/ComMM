# coding: utf-8
# @email: enoche.chow@gmail.com

"""
Main entry
# UPDATED: 2022-Feb-15
##########################
"""

import argparse
import os

os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model', '-m',
        choices=('PGL', 'PGL_MASKED', 'FREEDOM', 'FREEDOM_MASKED'),
        default='PGL', help='model to train',
    )
    parser.add_argument('--dataset', '-d', default='baby', help='dataset config name')
    parser.add_argument(
        '--data-path', type=os.path.abspath,
        help='directory containing dataset folders (defaults to ComMM/data)',
    )
    parser.add_argument('--gpu-id', type=int, default=0)
    parser.add_argument(
        '--cpu', action='store_true', help='force CPU execution',
    )
    parser.add_argument('--epochs', type=int, help='override the configured epochs')
    parser.add_argument('--mg', action="store_true", help='whether to use Mirror Gradient, default is False')
    parser.add_argument(
        '--no-save-model', action='store_true',
        help='do not save the best checkpoint',
    )

    args = parser.parse_args()
    config_dict = {'gpu_id': args.gpu_id, 'use_gpu': not args.cpu}
    if args.data_path is not None:
        config_dict['data_path'] = args.data_path
    if args.epochs is not None:
        config_dict['epochs'] = args.epochs

    try:
        from utils.quick_start import quick_start
    except ModuleNotFoundError as error:
        parser.error(
            'missing runtime dependency {!r}; install requirements.txt first'
            .format(error.name)
        )

    quick_start(
        model=args.model,
        dataset=args.dataset,
        config_dict=config_dict,
        save_model=not args.no_save_model,
        mg=args.mg,
    )


