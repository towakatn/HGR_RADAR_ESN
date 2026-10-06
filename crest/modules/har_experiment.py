#!/usr/bin/env python3
"""Compare reservoir topology and fusion stage on 16-bit HAR data in one room and distance."""

import argparse
from collections import Counter
from copy import deepcopy
import csv
from datetime import datetime
import json
from numbers import Integral
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np

DATASET_DIR = Path(__file__).resolve().parents[1] / 'HAR-Dataset-Project'

from .fusion_evaluation import (
    run_fusion_comparison, save_fusion_results, save_fusion_plot,
)
from .har_data import HARDataLoader, HAR_DISTANCE_METERS
from .har_config import DATA_CONFIG, FUSION_EXPERIMENT_CONFIG


HAR_PROTOCOLS = ('50_50', '10fold', 'loso')
DISTANCE_METERS = HAR_DISTANCE_METERS


def _counts(values):
    return {str(key): int(count) for key, count in sorted(Counter(values).items())}


def _distribution(y, metadata, indices):
    """Report the actual acquisition conditions represented in a partition."""
    selected = [metadata[index] for index in indices]
    return {
        'n_samples': len(selected),
        'classes': _counts(int(y[index]) for index in indices),
        **{key: _counts(row[key] for row in selected)
           for key in ('room', 'subject', 'action', 'distance')
           if all(key in row for row in selected)},
    }


def _save_split_manifest(results, metadata, destination):
    """Make train/test membership auditable by original recording filename."""
    fields = ['protocol', 'fold', 'partition', 'sample_index', 'filename',
              'room', 'subject', 'action', 'distance', 'repeat']
    with (destination / 'split_manifest.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for split in results['splits']:
            for partition in ('train', 'test'):
                for index in split[f'{partition}_indices']:
                    sample = metadata[index]
                    row = {key: sample.get(key) for key in fields if key in sample}
                    row.update(protocol=split['protocol'], fold=split['fold'],
                               partition=partition, sample_index=index,
                               repeat=sample.get('repeat', sample.get('repetition')))
                    writer.writerow(row)


def main(data_config=None, fusion_config=None, loaded_data=None):
    """Load one room and one distance, then persist a paired comparison and manifest.

    ``loaded_data`` can supply the loader's ``(maps, y, metadata)`` tuple for
    callers that already loaded the recordings. No network download is made.
    """
    data = deepcopy(DATA_CONFIG)
    if data_config is not None:
        data.update(data_config)
    config = deepcopy(FUSION_EXPERIMENT_CONFIG)
    if fusion_config is not None:
        config.update(fusion_config)
    output_dir = config.pop('output_dir', None)
    if (not isinstance(data['room'], Integral) or isinstance(data['room'], bool)
            or data['room'] < 1):
        raise ValueError('room must be one positive integer, selecting a single H identifier')
    room = int(data['room'])
    if (not isinstance(data['distance'], Integral) or isinstance(data['distance'], bool)
            or data['distance'] not in DISTANCE_METERS):
        raise ValueError('distance must be one integer in (1, 2, 3), selecting a single D identifier')
    distance = int(data['distance'])
    protocols = tuple(config['protocols'])
    if not protocols or set(protocols) - set(HAR_PROTOCOLS):
        raise ValueError(f'HAR protocols must be members of {HAR_PROTOCOLS}; '
                         'repetition numbers are not recording sessions')
    params = dict(config.get('reservoir_params') or {})
    if params.get('bias_scaling', 0.0) != 0.0:
        raise ValueError('HAR comparisons require bias_scaling=0.0')
    params['bias_scaling'] = 0.0
    config['reservoir_params'] = params
    for candidate in config.get('parameter_candidates') or []:
        if candidate.get('bias_scaling', 0.0) != 0.0:
            raise ValueError('HAR comparison candidates require bias_scaling=0.0')

    manifest = None
    if loaded_data is None:
        loader = HARDataLoader(data['base_dir'], room=room, distance=distance,
                               channels=data['channels'])
        loaded_data = loader.load_all_data()
        manifest = loader.data_manifest
    maps, y, metadata = loaded_data
    y = np.asarray(y)
    if not len(metadata) or any(row.get('room') != room for row in metadata):
        raise ValueError('Every selected recording must come from the single requested room')
    if any(row.get('distance') != distance for row in metadata):
        raise ValueError('Every selected recording must come from the single requested distance')
    if len(y) != len(metadata):
        raise ValueError('Labels and acquisition metadata must align')
    source_manifest_path = Path(data['base_dir']) / 'results' / f'download_H{room}_manifest.json'
    source_manifest = None
    if source_manifest_path.exists():
        with source_manifest_path.open(encoding='utf-8') as handle:
            source_manifest = json.load(handle)
        if source_manifest.get('room') != room:
            raise ValueError('Download provenance must identify the selected room')

    print(f'\nHAR 16-bit within-room comparison (existing RR_L, no bias): '
          f'room H{room}, distance D{distance} ({DISTANCE_METERS[distance]} m), '
          f'{len(y)} samples, {len(maps)} maps, '
          f"N={config['total_nodes']}, seeds={config['seeds']}", flush=True)
    results = run_fusion_comparison(maps, y, metadata, **config)
    results['configuration'].update(
        dataset='HAR-mmWave 16-bit', data=data,
        session_identifier=None,
        room=room, distance=distance,
        distance_meters=DISTANCE_METERS[distance],
        evaluation_scope='within one room and one distance; sample-level stratified 50:50 split '
                         'when protocol is 50_50; subjects may occur in both partitions',
        data_manifest=manifest,
        sample_filenames=[row.get('filename') for row in metadata],
        sample_metadata=metadata,
    )
    if source_manifest is not None:
        results['configuration'].update(
            source_commit=source_manifest.get('source_commit'),
            source_download_manifest=source_manifest,
        )
    results['dataset_summary'] = _distribution(y, metadata, range(len(y)))
    results['dataset_summary']['distance_meters'] = DISTANCE_METERS[distance]
    results['dataset_summary']['map_shapes'] = {
        name: list(np.asarray(sequences[0]).shape) for name, sequences in maps.items()
    }
    for split in results['splits']:
        split['train_distribution'] = _distribution(y, metadata, split['train_indices'])
        split['test_distribution'] = _distribution(y, metadata, split['test_indices'])
    if output_dir is None:
        timestamp = datetime.now(ZoneInfo('Asia/Tokyo')).strftime('%Y%m%d_%H%M%S_%f')
        output_dir = DATASET_DIR / 'results' / f'fusion_room{room}_distance{distance}_{timestamp}'
    destination = save_fusion_results(results, output_dir)
    _save_split_manifest(results, metadata, destination)
    with (destination / 'dataset_summary.json').open('w', encoding='utf-8') as handle:
        json.dump(results['dataset_summary'], handle, ensure_ascii=False, indent=2, allow_nan=False)
    save_fusion_plot(results, destination)
    print(f"\n{'Protocol':<15} {'Method':<32} {'Accuracy (mean +/- fold/seed std)':>32}")
    for row in results['summary']:
        print(f"{row['protocol']:<15} {row['method']:<32} "
              f"{row['mean_accuracy'] * 100:7.2f}% +/- {row['std_accuracy'] * 100:6.2f}%")
    print(f'Results: {destination.resolve()}', flush=True)
    return results


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--room', type=int, help='use only this filename H identifier; default 1')
    parser.add_argument('--distance', type=int, choices=(1, 2, 3),
                        help='use only this filename D identifier; default 1 (1.5 m)')
    parser.add_argument('--base-dir', type=Path, help='HAR repository root containing the 16-bit folder')
    parser.add_argument('--total-nodes', type=int, help='same total N for all methods; default 400')
    parser.add_argument('--regularization', type=float,
                        help=f"existing RR_L lambda; default {FUSION_EXPERIMENT_CONFIG['regularization']}")
    parser.add_argument('--seeds', help='comma-separated reservoir seeds; default 42,43,44')
    parser.add_argument('--protocols', help='comma-separated: 50_50,10fold,loso; default 50_50')
    parser.add_argument('--n-splits', type=int, help='stratified CV fold count; default 10')
    parser.add_argument('--output-dir', type=Path, help='directory for JSON, CSV, and plot outputs')
    return parser.parse_args(argv)


def cli(argv=None):
    args = parse_args(argv)
    data, config = deepcopy(DATA_CONFIG), deepcopy(FUSION_EXPERIMENT_CONFIG)
    if args.room is not None:
        data['room'] = args.room
    if args.distance is not None:
        data['distance'] = args.distance
    if args.base_dir is not None:
        data['base_dir'] = str(args.base_dir.resolve())
    for name in ('total_nodes', 'regularization', 'n_splits'):
        value = getattr(args, name)
        if value is not None:
            config[name] = value
    if args.seeds is not None:
        config['seeds'] = [int(value.strip()) for value in args.seeds.split(',')]
    if args.protocols is not None:
        config['protocols'] = [value.strip() for value in args.protocols.split(',')]
    if args.output_dir is not None:
        config['output_dir'] = str(args.output_dir.resolve())
    return main(data, config)


if __name__ == '__main__':
    cli()
