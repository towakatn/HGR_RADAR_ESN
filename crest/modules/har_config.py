"""Within-room HAR experiments using the high-precision radar recordings."""

from pathlib import Path


DATASET_DIR = Path(__file__).resolve().parents[1] / 'HAR-Dataset-Project'

DATA_CONFIG = {
    'base_dir': str(DATASET_DIR),
    'room': 1,
    'distance': 1,
    'channels': [0, 1, 2, 3],
}

FUSION_EXPERIMENT_CONFIG = {
    'total_nodes': 400,
    'regularization': 0.1,  # Best mean inner-validation score in the H1/D1 diagnostic.
    'reservoir_params': {
        'spectral_radius': 0.95,
        'input_scaling': 0.2,
        'density': 0.1,
        'leakage_rate': 0.05,
        'bias_scaling': 0.0,
        'temperature': 1.0,
        'standardize_inputs': True,
    },
    'seeds': [42, 43, 44],
    'protocols': ['50_50'],
    'n_splits': 10,
    'split_seed': 42,
    'n_trials': 1,
    'parameter_candidates': None,
    'late_fusion_methods': ['mean', 'product', 'geometric', 'max'],
    'inner_validation_size': 0.25,
    'output_dir': None,
}
