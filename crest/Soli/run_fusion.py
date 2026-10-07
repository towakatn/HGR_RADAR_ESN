#!/usr/bin/env python3
"""Run paired Soli experiments for reservoir topology and fusion stage."""

import argparse
from datetime import datetime
from pathlib import Path
import json
import sys
from zoneinfo import ZoneInfo

DATASET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATASET_DIR))
sys.path.insert(0, str(DATASET_DIR.parent))

from modules.data_loaders import DualDataTypeLoader
from modules.fusion_evaluation import (
    build_soli_maps, run_fusion_comparison, save_fusion_results, save_fusion_plot,
)
from soli_config import DATA_CONFIG, FUSION_EXPERIMENT_CONFIG


def main(data_config=None, fusion_config=None, loaded_data=None):
    """Load once, run the controlled suite, and persist paired measurements."""
    data = dict(DATA_CONFIG if data_config is None else data_config)
    config = dict(FUSION_EXPERIMENT_CONFIG if fusion_config is None else fusion_config)
    output_dir = config.pop('output_dir', None)
    if loaded_data is None:
        loader = DualDataTypeLoader(channels=data['channels'], base_dir=data['base_dir'])
        loaded_data = loader.load_gesture_data(
            max_samples_per_gesture_subject=data['max_samples_per_gesture_subject'])
    X_md, X_rtm, y, metadata = loaded_data
    maps = build_soli_maps(X_md, X_rtm, channels=data['channels'])
    print(f"\nControlled fusion comparison (existing RR_L): {len(y)} samples, {len(maps)} maps, "
          f"N={config['total_nodes']}, seeds={config['seeds']}", flush=True)
    results = run_fusion_comparison(maps, y, metadata, **config)
    results['configuration']['data'] = data
    results['configuration']['sample_filenames'] = [row.get('filename') for row in metadata]
    if output_dir is None:
        timestamp = datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S_%f')
        output_dir = DATASET_DIR / 'results' / f'fusion_{timestamp}'
    destination = save_fusion_results(results, output_dir)
    save_fusion_plot(results, destination)
    print(f"\n{'Protocol':<15} {'Method':<32} {'Accuracy (mean +/- fold/seed std)':>32}")
    for row in results['summary']:
        print(f"{row['protocol']:<15} {row['method']:<32} "
              f"{row['mean_accuracy'] * 100:7.2f}% +/- {row['std_accuracy'] * 100:6.2f}%")
    print(f"Results: {destination.resolve()}", flush=True)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--quick', action='store_true', help='32 nodes, seed 42, two sample indices, 50:50 only')
    parser.add_argument('--total-nodes', type=int, help='same N for every method; must divide map count')
    parser.add_argument('--regularization', type=float, help='lambda * I in the existing RR_L ridge readout')
    parser.add_argument('--seeds', help='comma-separated reservoir seeds, e.g. 42,43,44')
    parser.add_argument('--protocols', help='comma-separated: 50_50,10fold,session_split,loso')
    parser.add_argument('--n-splits', type=int, help='stratified CV fold count')
    parser.add_argument('--search', choices=('fixed', 'shared_candidates', 'bayesian'),
                        help='search strategy; default bayesian (quick: fixed)')
    parser.add_argument('--trials', type=int,
                        help='same evaluations per method; default 60 for Bayesian, 1 for fixed')
    parser.add_argument('--candidates', type=Path,
                        help='JSON list for shared_candidates; selects that mode unless --search is set')
    parser.add_argument('--max-samples', type=int, help='select filename sample index < this value per gesture/subject')
    parser.add_argument('--base-dir', type=Path, help='directory containing Soli DTM/RTM')
    parser.add_argument('--output-dir', type=Path, help='save JSON/CSV reports here')
    return parser.parse_args(argv)


def cli(argv=None):
    args = parse_args(argv)
    data, config = dict(DATA_CONFIG), dict(FUSION_EXPERIMENT_CONFIG)
    if args.quick:
        data['max_samples_per_gesture_subject'] = 2
        config.update(total_nodes=32, seeds=[42], protocols=['50_50'],
                      search_strategy='fixed', n_trials=1, parameter_candidates=None)
    for arg_name, config_name in [('total_nodes', 'total_nodes'), ('regularization', 'regularization'),
                                  ('n_splits', 'n_splits')]:
        value = getattr(args, arg_name)
        if value is not None:
            config[config_name] = value
    if args.seeds is not None:
        config['seeds'] = [int(value.strip()) for value in args.seeds.split(',')]
    if args.protocols is not None:
        config['protocols'] = [value.strip() for value in args.protocols.split(',')]
    if args.candidates is not None:
        with args.candidates.open(encoding='utf-8') as handle:
            config['parameter_candidates'] = json.load(handle)
        if args.search is None:
            config['search_strategy'] = 'shared_candidates'
        if args.trials is None:
            config['n_trials'] = len(config['parameter_candidates'])
    if args.search is not None:
        config['search_strategy'] = args.search
        if args.trials is None:
            if args.search == 'fixed':
                config['n_trials'] = 1
            elif args.search == 'bayesian':
                config['n_trials'] = 60
    if args.trials is not None:
        config['n_trials'] = args.trials
    if args.max_samples is not None:
        if args.max_samples < 1:
            raise ValueError('--max-samples must be positive')
        data['max_samples_per_gesture_subject'] = args.max_samples
    if args.base_dir is not None:
        data['base_dir'] = str(args.base_dir.resolve())
    if args.output_dir is not None:
        config['output_dir'] = str(args.output_dir.resolve())
    return main(data, config)


if __name__ == '__main__':
    cli()
