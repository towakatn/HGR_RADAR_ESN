"""Numerical regressions recorded from the original separate Soli/Dop-NET modules.

Run from crest with: OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests
The synthetic fixtures require no downloaded radar data and exercise variable lengths.
"""

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.converters import SeparateChannelConverter, SeparateChannelRTMConverter
from modules.classifiers import classifier_factory
from modules import evaluation as evaluation_module
from modules.data_loaders import DualDataTypeLoader, RCDataLoader
from modules.evaluation import run_dopnet_evaluation, run_soli_evaluation
from modules.readouts import ClassifierESNReadout, FeatESNReadout, SingleReservoirESN
from modules.reservoir import VariableLengthESN
from modules.reservoir_computer import ReservoirComputer, prepare_rc_input


RESERVOIR_PARAMS = dict(
    n_reservoir=6, spectral_radius=0.82, input_scaling=0.7,
    density=0.65, leakage_rate=0.22, bias_scaling=0.1, random_state=17,
)
CLASSIFIER_PARAMS = {
    'rf': dict(n_estimators=7, max_depth=3, random_state=17, n_jobs=1),
    'svm': dict(kernel='rbf', C=2.0, gamma='scale', random_state=17),
    'ridge': dict(alpha=0.4, random_state=17),
}


def synthetic_data():
    """Two subjects, six sessions, two gestures, four aligned variable-length channels."""
    rng = np.random.RandomState(2023)
    md = {ch: [] for ch in range(4)}
    rtm = {ch: [] for ch in range(4)}
    labels, metadata = [], []
    for subject in range(2):
        for session in range(6):
            for label in range(2):
                frames = 3 + len(labels) % 4
                labels.append(label)
                metadata.append(dict(subject=subject, session=session, gesture=label))
                for ch in range(4):
                    md[ch].append(rng.normal(0.15 * label, 0.8, (frames, 2)))
                    rtm[ch].append(rng.normal(-0.1 * label, 0.8, (frames, 3)))
    return md, rtm, np.array(labels), metadata


def soli_methods():
    for nonlinear, name in [('none', 'multi_rr_l'), ('square_tanh', 'multi_rr_n')]:
        params = RESERVOIR_PARAMS.copy()
        params['n_reservoir_per_stream'] = params.pop('n_reservoir')
        params.update(n_selected_nodes=3, regularization=0.03, nonlinear_features=nonlinear)
        yield name, FeatESNReadout, params
    for classifier in ['rf', 'svm']:
        yield 'multi_' + classifier, ClassifierESNReadout, dict(
            RESERVOIR_PARAMS, classifier_type=classifier,
            classifier_config=CLASSIFIER_PARAMS[classifier],
        )
    for classifier in ['rf', 'svm', 'ridge']:
        yield 'single_' + classifier, SingleReservoirESN, dict(
            RESERVOIR_PARAMS, channels=[0, 1, 2, 3], node_selection_ratio=0.5,
            classifier_type=classifier, classifier_config=CLASSIFIER_PARAMS[classifier],
        )


def dopnet_classifiers():
    return [
        ('RF', classifier_factory('rf', dict(n_estimators=7, max_depth=3, n_jobs=1))),
        ('SVM', classifier_factory('svm', dict(kernel='rbf', C=2.0, gamma='scale'))),
        ('Ridge', classifier_factory('ridge', dict(alpha=0.4), use_random_state=False)),
    ]


def reservoir_results():
    md, _, y, _ = synthetic_data()
    esn = VariableLengthESN(**RESERVOIR_PARAMS)
    states = esn.transform_sequences(md[0])
    dop_params = {key: value for key, value in RESERVOIR_PARAMS.items() if key != 'bias_scaling'}
    rc = ReservoirComputer(**dop_params)
    rc.fit(md[0], y)
    return dict(
        soli_input_weights=esn.W_in, soli_reservoir_weights=esn.W_res,
        soli_bias=esn.W_bias, soli_states=states,
        dop_input_weights=rc.W_input, dop_reservoir_weights=rc.W_reservoir.toarray(),
        dop_states=rc.states, dop_transformed=rc.transform(md[0][::-1]),
    )


def soli_results(name, model_class, params):
    md, rtm, y, metadata = synthetic_data()
    model = model_class(**params)
    model.fit(md, rtm, y)
    prediction, _, _ = model.predict(md, rtm)
    if isinstance(model, FeatESNReadout):
        features = model._extract_reservoir_states(md, rtm)
        extra = dict(readout_weights=model.W_out,
                     selected_md=[model.selected_nodes_md[ch] for ch in range(4)],
                     selected_rtm=[model.selected_nodes_rtm[ch] for ch in range(4)])
    else:
        features = model.extract_features(md, rtm)
        extra = {}
        if isinstance(model, SingleReservoirESN):
            extra['selected_nodes'] = model.selected_indices
    with contextlib.redirect_stdout(io.StringIO()):
        evaluation = run_soli_evaluation(model_class, md, rtm, y, metadata, params, name)
    return dict(features=features[:3], prediction=prediction, evaluation=evaluation, **extra)


def dopnet_results():
    md, _, y, soli_metadata = synthetic_data()
    metadata = [dict(person=['A', 'B'][m['subject']], sample_idx=m['session']) for m in soli_metadata]
    params = {key: value for key, value in RESERVOIR_PARAMS.items() if key != 'bias_scaling'}
    rc = ReservoirComputer(**params)
    rc.fit(md[0], y)
    predictions = {}
    for name, create in dopnet_classifiers():
        clf = create(params['random_state'])
        clf.fit(rc.states, y)
        predictions[name] = clf.predict(rc.states)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluation = run_dopnet_evaluation(md[0], y, metadata, params, dopnet_classifiers())
    return dict(predictions=predictions, evaluation=evaluation)


def split_traces():
    """Record exact sample membership, direction, and Dop-NET classifier seeds."""
    _, _, y, metadata = synthetic_data()
    ids = {ch: [np.array([[i]]) for i in range(len(y))] for ch in range(4)}
    events = []

    class RecordingSoliModel:
        def fit(self, md, rtm, labels, verbose=False):
            events.append(dict(train=[int(x[0, 0]) for x in md[0]]))
            return 0.0, 0.0

        def predict(self, md, rtm, verbose=False):
            events[-1]['test'] = [int(x[0, 0]) for x in md[0]]
            return np.zeros(len(md[0]), dtype=int), 0.0, 0.0

    with contextlib.redirect_stdout(io.StringIO()):
        run_soli_evaluation(RecordingSoliModel, ids, ids, y, metadata, {}, 'trace')
    soli = events.copy()
    events.clear()

    class RecordingDopClassifier:
        def fit(self, states, labels):
            events[-1]['train'] = states[:, 0].astype(int).tolist()

        def predict(self, states):
            events[-1]['test'] = states[:, 0].astype(int).tolist()
            return np.zeros(len(states), dtype=int)

    def create(seed):
        events.append(dict(seed=seed))
        return RecordingDopClassifier()

    # Replace only the expensive feature calculation with identifiers. The actual
    # evaluation functions perform all splitting and classifier seed selection.
    def fold_states(train, test, config, fold_num):
        return np.array([x[0] for x in train]), np.array([x[0] for x in test])

    dop_metadata = [dict(person=['A', 'B'][m['subject']], sample_idx=m['session']) for m in metadata]
    with patch.dict(run_dopnet_evaluation.__globals__, {
        '_compute_all_states': lambda samples, params: np.arange(len(samples)).reshape(-1, 1),
    }), patch.dict(evaluation_module.evaluate_10fold.__globals__, {
        '_compute_fold_states': fold_states,
    }), contextlib.redirect_stdout(io.StringIO()):
        run_dopnet_evaluation(ids[0], y, dop_metadata, {'random_state': 17}, [('Trace', create)])
    return dict(soli=soli, dopnet=events)


class NumericalRegressionTests(unittest.TestCase):
    def assert_nested_close(self, actual, expected):
        if isinstance(expected, dict):
            self.assertEqual(set(actual), set(expected))
            for key, value in expected.items():
                with self.subTest(field=key):
                    self.assert_nested_close(actual[key], value)
        else:
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-12)

    def test_reservoir_weights_and_variable_length_states(self):
        self.assert_nested_close(reservoir_results(), EXPECTED['reservoirs'])

    def test_all_seven_soli_readouts_and_full_evaluations(self):
        for name, model_class, params in soli_methods():
            with self.subTest(method=name):
                self.assert_nested_close(soli_results(name, model_class, params), EXPECTED[name])

    def test_dopnet_three_classifiers_and_full_evaluation(self):
        self.assert_nested_close(dopnet_results(), EXPECTED['dopnet'])

    def test_exact_split_membership_direction_and_fold_seeds(self):
        self.assertEqual(split_traces(), EXPECTED['splits'])

    def test_soli_input_scaling_preserves_original_behavior(self):
        # The original VariableLengthESN stores input_scaling but does not apply it.
        # A structural refactor must not silently alter that numerical behavior.
        md, _, _, _ = synthetic_data()
        low = VariableLengthESN(**dict(RESERVOIR_PARAMS, input_scaling=0.01))
        low_states = low.transform_sequences(md[0])
        high = VariableLengthESN(**dict(RESERVOIR_PARAMS, input_scaling=100.0))
        np.testing.assert_array_equal(low_states, high.transform_sequences(md[0]))

    def test_dopnet_randomness_is_independent_of_global_numpy_seed(self):
        md, _, y, _ = synthetic_data()
        params = {key: value for key, value in RESERVOIR_PARAMS.items() if key != 'bias_scaling'}
        np.random.seed(1)
        first = ReservoirComputer(**params).fit(md[0], y)
        np.random.seed(999)
        second = ReservoirComputer(**params).fit(md[0], y)
        np.testing.assert_array_equal(first.states, second.states)


def load_entrypoint(dataset):
    path = Path(__file__).resolve().parents[1] / dataset / 'run_all.py'
    spec = importlib.util.spec_from_file_location('crest_' + dataset.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    original_path = sys.path.copy()
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return module


class EntrypointTests(unittest.TestCase):
    def test_soli_entrypoint_forwards_dataset_and_reservoir_arguments(self):
        entry = load_entrypoint('Soli')
        md, rtm, y, metadata = synthetic_data()
        data = dict(base_dir='/tmp/soli-argument-fixture', channels=[0, 1, 2, 3],
                    max_samples_per_gesture_subject=7)
        multi = RESERVOIR_PARAMS.copy()
        single = dict(RESERVOIR_PARAMS, n_reservoir=8, random_state=23,
                      node_selection_ratio=0.5)
        original_multi, original_single = multi.copy(), single.copy()
        evaluations = []

        def evaluate(model_class, actual_md, actual_rtm, labels, actual_metadata, params, name):
            self.assertIs(actual_md, md)
            self.assertIs(actual_rtm, rtm)
            self.assertIs(labels, y)
            self.assertIs(actual_metadata, metadata)
            # Constructing the real class catches invalid or missing forwarded keys.
            model = model_class(**params)
            evaluations.append((model, params))
            return EXPECTED[name.lower()]['evaluation']

        with patch.object(entry, 'DualDataTypeLoader') as loader, \
                patch.object(entry, 'run_soli_evaluation', side_effect=evaluate), \
                contextlib.redirect_stdout(io.StringIO()):
            loader.return_value.load_gesture_data.return_value = md, rtm, y, metadata
            results = entry.main(data, multi, single, include_fusion=False)
            loader.assert_called_once_with(channels=data['channels'], base_dir=data['base_dir'])
            loader.return_value.load_gesture_data.assert_called_once_with(
                max_samples_per_gesture_subject=7)

        self.assertEqual(list(results), ['Multi_RR_L', 'Multi_RR_N', 'Multi_SVM', 'Multi_RF',
                                         'Single_RF', 'Single_SVM', 'Single_Ridge'])
        self.assertEqual(multi, original_multi)
        self.assertEqual(single, original_single)
        for index, (model, params) in enumerate(evaluations):
            if index < 2:
                self.assertIsInstance(model, FeatESNReadout)
                self.assertEqual(params['n_reservoir_per_stream'], 6)
                self.assertEqual(params['n_selected_nodes'], 6)
                self.assertEqual(params['random_state'], 17)
                self.assertEqual(params['nonlinear_features'], ['none', 'square_tanh'][index])
            elif index < 4:
                self.assertIsInstance(model, ClassifierESNReadout)
                self.assertEqual(model.n_reservoir, 6)
                self.assertEqual(model.random_state, 17)
                self.assertEqual(model.classifier_type, ['svm', 'rf'][index - 2])
            else:
                self.assertIsInstance(model, SingleReservoirESN)
                self.assertEqual(model.n_reservoir, 8)
                self.assertEqual(model.random_state, 23)
                self.assertEqual(model.n_selected_nodes, 4)
                self.assertEqual(model.classifier_type, ['rf', 'svm', 'ridge'][index - 4])
            config = multi if index < 4 else single
            for key in ['spectral_radius', 'input_scaling', 'density', 'leakage_rate', 'bias_scaling']:
                self.assertEqual(getattr(model, key), config[key])

    def test_dopnet_entrypoint_transposes_data_and_forwards_classifier_seeds(self):
        entry = load_entrypoint('Dop-NET')
        signals = [np.arange(6).reshape(2, 3), np.arange(8).reshape(2, 4)]
        labels = [0, 1]
        metadata = [dict(person='A', sample_idx=0), dict(person='A', sample_idx=1)]
        params = {key: value for key, value in RESERVOIR_PARAMS.items() if key != 'bias_scaling'}
        data = dict(data_dir='/tmp/dopnet-argument-fixture')
        expected = EXPECTED['dopnet']['evaluation']

        def evaluate(prepared, actual_labels, actual_metadata, actual_params, classifiers):
            for original, sequence in zip(signals, prepared):
                np.testing.assert_array_equal(sequence, original.T)
            np.testing.assert_array_equal(actual_labels, labels)
            self.assertIs(actual_metadata, metadata)
            self.assertIs(actual_params, params)
            self.assertEqual([name for name, _ in classifiers], ['RF', 'SVM', 'Ridge'])
            rf, svm, ridge = [create(31) for _, create in classifiers]
            self.assertEqual(rf.n_estimators, entry.RF_CONFIG['n_estimators'])
            self.assertEqual(rf.n_jobs, entry.RF_CONFIG['n_jobs'])
            self.assertEqual(rf.random_state, 31)
            self.assertEqual(svm.kernel, entry.SVM_CONFIG['kernel'])
            self.assertEqual(svm.C, entry.SVM_CONFIG['C'])
            self.assertEqual(svm.gamma, entry.SVM_CONFIG['gamma'])
            self.assertEqual(svm.random_state, 31)
            self.assertEqual(ridge.alpha, entry.RIDGE_CONFIG['alpha'])
            self.assertIsNone(ridge.random_state)
            return expected

        with patch.object(entry, 'RCDataLoader') as loader, \
                patch.object(entry, 'run_dopnet_evaluation', side_effect=evaluate), \
                contextlib.redirect_stdout(io.StringIO()):
            loader.return_value.load_all_data.return_value = signals, labels, metadata
            self.assertEqual(entry.main(data, params), expected)
            loader.assert_called_once_with(data_dir=data['data_dir'])
            loader.return_value.load_all_data.assert_called_once_with()


class LoaderAndConverterTests(unittest.TestCase):
    def test_dopnet_amplitude_and_axis_order_preserve_magnitude(self):
        signal = np.array([[3 + 4j, -5j, 12j], [-8 + 6j, 0j, 1 + 1j]])
        loader = RCDataLoader()
        amplitude = loader.convert_to_normalized_spectrogram(signal)
        np.testing.assert_allclose(amplitude, [[5, 5, 12], [10, 0, np.sqrt(2)]])
        prepared = prepare_rc_input([amplitude, amplitude[:, :2]])
        np.testing.assert_array_equal(prepared[0], amplitude.T)
        self.assertEqual([item.shape for item in prepared], [(3, 2), (2, 2)])

    def test_conversion_sums_axes_and_keeps_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            raw = np.arange(2 * 32 * 32).reshape(2, 32, 32)
            source = base / '0_1_0.h5'
            with h5py.File(source, 'w') as handle:
                for ch in range(4):
                    handle[f'ch{ch}'] = (raw + ch).reshape(2, 1024)
                handle['label'] = 0
                handle['timestamp'] = [1.0, 2.0]
            dtm = SeparateChannelConverter(input_dir=str(base), base_output_dir=str(base / 'DTM'))
            rtm = SeparateChannelRTMConverter(input_dir=str(base), base_output_dir=str(base / 'RTM'))
            for converter, key, axis in [(dtm, 'rd_evolution', 1), (rtm, 'rtm', 2)]:
                results = converter.process_single_file(str(source))
                for ch in range(4):
                    with h5py.File(results[ch], 'r') as handle:
                        np.testing.assert_array_equal(handle[key][:], (raw + ch).sum(axis=axis))
                        np.testing.assert_allclose(handle['time_axis'][:], [0, 0.05])
                        self.assertEqual(handle['metadata/label'][()], 0)
                        np.testing.assert_array_equal(handle['metadata/timestamp'][:], [1, 2])
                        self.assertEqual(handle.attrs['channel'], ch)
                result = (converter.extract_range_doppler_evolution(raw)
                          if key == 'rd_evolution' else converter.extract_range_time_map(raw))
                np.testing.assert_array_equal(result[0], raw.sum(axis=axis))

    def test_soli_loader_filter_order_and_missing_channel_handling(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            # Original sorting is lexical; sessions 10 and 2 appear in that order.
            filenames = ['1_0_2', '0_10_0', '0_0_25', '0_0_3', '0_0_2', '0_0_10']
            for ch in range(4):
                for data_type, prefix, key in [('DTM', 'rde', 'rd_evolution'), ('RTM', 'rtm', 'rtm')]:
                    directory = base / data_type / f'{ch}ch_{data_type}Data'
                    directory.mkdir(parents=True)
                    for filename in filenames:
                        if filename == '0_0_3' and ch == 2 and data_type == 'RTM':
                            continue
                        with h5py.File(directory / f'{prefix}_ch{ch}_{filename}.h5', 'w') as handle:
                            handle[key] = np.full((3, 2), ch + int(filename.split('_')[2]))
            md, rtm, y, metadata = DualDataTypeLoader(base_dir=tmpdir).load_gesture_data()
            np.testing.assert_array_equal(y, [0, 0, 1])
            self.assertEqual([m['filename'] for m in metadata], ['0_0_10.h5', '0_0_2.h5', '1_0_2.h5'])
            self.assertEqual([m['session'] for m in metadata], [10, 2, 2])
            for ch in range(4):
                np.testing.assert_array_equal(md[ch][0], np.full((3, 2), ch + 10))
                np.testing.assert_array_equal(rtm[ch][1], np.full((3, 2), ch + 2))


# These goldens were captured before the refactor, from the original modules,
# using the fixture and explicit parameters above. Wall-clock timings are omitted.
EXPECTED = json.loads((Path(__file__).parent / 'fixtures' / 'refactor_expected.json').read_text())


if __name__ == '__main__':
    unittest.main()
