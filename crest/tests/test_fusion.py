"""Checks for the controlled reservoir/fusion experiment, using synthetic sequences.

Run from crest: OPENBLAS_NUM_THREADS=1 .venv/bin/python -m unittest discover -s tests
"""

import contextlib
import csv
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.fusion import FusionESN, fuse_probabilities
from modules.readouts import FeatESNReadout, fit_ridge_readout
from modules.fusion_evaluation import (
    build_soli_maps, make_soli_splits, run_fusion_comparison, save_fusion_results,
)


ARCHITECTURES = (
    "single_map", "single_early", "parallel_early",
    "parallel_intermediate", "parallel_late",
)
PARAMETERS = dict(
    total_nodes=12, regularization=0.17, spectral_radius=0.75,
    input_scaling=0.6, density=0.7, leakage_rate=0.3,
    bias_scaling=0.1, random_state=19,
)


def synthetic_maps():
    """Aligned maps with unequal feature widths and variable sample durations."""
    rng = np.random.default_rng(2031)
    labels = np.tile([11, 27, 4], 6)
    maps = {"md": [], "rtm": [], "angle": []}
    for sample, label in enumerate(labels):
        frames = 3 + sample % 5
        for name, width in [("md", 2), ("rtm", 3), ("angle", 1)]:
            maps[name].append(rng.normal(label / 20, 0.8, (frames, width)))
    return maps, labels


def make_model(architecture, **overrides):
    parameters = dict(PARAMETERS, **overrides)
    if architecture == "single_map":
        parameters.setdefault("map_name", "md")
    return FusionESN(architecture, **parameters)


def softmax(logits):
    centered = logits - np.max(logits, axis=-1, keepdims=True)
    exponentials = np.exp(centered)
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def load_soli_entrypoint(filename):
    path = Path(__file__).resolve().parents[1] / "Soli" / filename
    spec = importlib.util.spec_from_file_location("fusion_test_" + path.stem, path)
    module = importlib.util.module_from_spec(spec)
    original_path = sys.path.copy()
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path[:] = original_path
    return module


class FusionArchitectureTests(unittest.TestCase):
    def setUp(self):
        self.maps, self.y = synthetic_maps()

    def test_all_architectures_default_to_no_reservoir_bias(self):
        from modules.fusion_evaluation import RESERVOIR_DEFAULTS
        config = load_soli_entrypoint("soli_config.py")
        self.assertEqual(RESERVOIR_DEFAULTS["bias_scaling"], 0.0)
        for name in ("MULTI_RESERVOIR_CONFIG", "SINGLE_RESERVOIR_CONFIG"):
            self.assertEqual(getattr(config, name)["bias_scaling"], 0.0)
        self.assertEqual(config.FUSION_EXPERIMENT_CONFIG["reservoir_params"].get("bias_scaling", 0.0), 0.0)
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                parameters = dict(PARAMETERS)
                parameters.pop("bias_scaling")
                if architecture == "single_map":
                    parameters["map_name"] = "md"
                model = FusionESN(architecture, **parameters).fit(self.maps, self.y)
                for reservoir in model.reservoirs_:
                    np.testing.assert_array_equal(reservoir.W_bias, 0.0)
                counts = model.parameter_counts
                self.assertEqual(counts["reservoir_biases"], 0)
                self.assertEqual(counts["fixed_parameters"],
                                 counts["input_weights"] + counts["recurrent_weights"])

    def test_exact_total_nodes_and_architecture_routing(self):
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = make_model(architecture).fit(self.maps, self.y)
                self.assertEqual(sum(model.node_counts_), PARAMETERS["total_nodes"])
                self.assertEqual(len(model.reservoirs_), len(model.node_counts_))
                self.assertEqual(model.extract_features(self.maps).shape, (len(self.y), 12))
                if architecture.startswith("single"):
                    self.assertEqual(list(model.node_counts_), [12])
                else:
                    self.assertEqual(list(model.node_counts_), [4, 4, 4])
                if architecture == "single_map":
                    expected_widths = [2]
                elif architecture.endswith("early"):
                    expected_widths = [6] * len(model.reservoirs_)
                else:
                    expected_widths = [self.maps[name][0].shape[1] for name in model.map_names_]
                self.assertEqual([r.W_in.shape[1] for r in model.reservoirs_], expected_widths)
                expected_readouts = 3 if architecture == "parallel_late" else 1
                self.assertEqual(len(model.readouts_), expected_readouts)
                self.assertEqual(sum(readout.size for readout in model.readouts_), 12 * 3)

    def test_single_map_ignores_other_map_values(self):
        model = make_model("single_map").fit(self.maps, self.y)
        changed = dict(self.maps)
        changed["rtm"] = [sequence * -100 + 1e5 for sequence in self.maps["rtm"]]
        changed["angle"] = [sequence * 500 for sequence in self.maps["angle"]]
        np.testing.assert_array_equal(model.extract_features(self.maps), model.extract_features(changed))
        np.testing.assert_array_equal(model.predict_proba(self.maps), model.predict_proba(changed))

    def test_intermediate_and_late_share_identical_reservoir_features(self):
        intermediate = make_model("parallel_intermediate").fit(self.maps, self.y)
        late = make_model("parallel_late").fit(self.maps, self.y)
        np.testing.assert_array_equal(
            intermediate.extract_features(self.maps), late.extract_features(self.maps),
        )
        for first, second in zip(intermediate.reservoirs_, late.reservoirs_):
            np.testing.assert_array_equal(first.W_in, second.W_in)
            np.testing.assert_array_equal(first.W_res, second.W_res)

    def test_parallel_architectures_share_recurrent_weights_despite_input_widths(self):
        models = [make_model(architecture).fit(self.maps, self.y)
                  for architecture in ARCHITECTURES[2:]]
        for reservoirs in zip(*(model.reservoirs_ for model in models)):
            for reservoir in reservoirs[1:]:
                np.testing.assert_array_equal(reservoir.W_res, reservoirs[0].W_res)
                np.testing.assert_array_equal(reservoir.W_bias, reservoirs[0].W_bias)

    def test_prediction_map_order_is_fixed_by_training_map_order(self):
        reordered = {name: self.maps[name] for name in reversed(self.maps)}
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = make_model(architecture).fit(self.maps, self.y)
                np.testing.assert_array_equal(
                    model.predict_proba(self.maps), model.predict_proba(reordered),
                )

    def test_map_specific_branches_do_not_receive_other_map_values(self):
        for architecture in ("parallel_intermediate", "parallel_late"):
            with self.subTest(architecture=architecture):
                model = make_model(architecture).fit(self.maps, self.y)
                original = model.extract_features(self.maps)
                changed = dict(self.maps)
                changed["rtm"] = [sequence + 100 for sequence in self.maps["rtm"]]
                transformed = model.extract_features(changed)
                offset = 0
                for name, nodes in zip(model.map_names_, model.node_counts_):
                    before = original[:, offset:offset + nodes]
                    after = transformed[:, offset:offset + nodes]
                    if name == "rtm":
                        self.assertFalse(np.allclose(before, after))
                    else:
                        np.testing.assert_array_equal(before, after)
                    offset += nodes

    def test_readouts_minimize_existing_ridge_sum_loss(self):
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = make_model(architecture).fit(self.maps, self.y)
                features = model.extract_features(self.maps)
                targets = (self.y[:, None] == model.classes_[None, :]).astype(float)
                branch_features = (
                    np.split(features, np.cumsum(model.node_counts_)[:-1], axis=1)
                    if architecture == "parallel_late" else [features]
                )
                for states, readout in zip(branch_features, model.readouts_):
                    # The original RR_L gradient is zero for ||XW-Y||² + lambda ||W||².
                    gradient = states.T @ (states @ readout - targets)
                    gradient += PARAMETERS["regularization"] * readout
                    np.testing.assert_allclose(gradient, 0, rtol=0, atol=1e-12)

    def test_readouts_match_existing_linear_ridge_regression(self):
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = make_model(architecture).fit(self.maps, self.y)
                features = model.extract_features(self.maps)
                branches = (np.split(features, np.cumsum(model.node_counts_)[:-1], axis=1)
                            if architecture == "parallel_late" else [features])
                for states, readout in zip(branches, model.readouts_):
                    # Match the contiguous branch arrays used during fitting;
                    # a strided split can select a different BLAS accumulation path.
                    states = np.ascontiguousarray(states)
                    # Run the existing RR_L fit on the exact same reservoir states.
                    legacy = FeatESNReadout.__new__(FeatESNReadout)
                    legacy.nonlinear_features = 'none'
                    legacy.regularization = model.regularization
                    legacy._extract_reservoir_states = lambda *args, states=states, **kwargs: states
                    legacy.fit(None, None, self.y)
                    np.testing.assert_array_equal(readout, legacy.W_out.T)

    def test_existing_ridge_preserves_pseudoinverse_fallback(self):
        features = np.array([[1., 1.], [2., 2.]])
        targets = np.eye(2)
        expected = np.dot(np.dot(np.linalg.pinv(features.T @ features), features.T), targets).T
        np.testing.assert_array_equal(fit_ridge_readout(features, targets, 0.), expected)

    def test_duplicated_samples_require_proportional_lambda_in_existing_ridge(self):
        repeated_maps = {name: sequences * 2 for name, sequences in self.maps.items()}
        repeated_y = np.tile(self.y, 2)
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                original = make_model(architecture).fit(self.maps, self.y)
                repeated = make_model(architecture, regularization=2 * PARAMETERS["regularization"]).fit(
                    repeated_maps, repeated_y)
                for first, second in zip(original.readouts_, repeated.readouts_):
                    np.testing.assert_allclose(first, second, rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(
                    original.predict_proba(self.maps), repeated.predict_proba(self.maps),
                    rtol=1e-10, atol=1e-12,
                )

    def test_probabilities_and_noncontiguous_class_labels(self):
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = make_model(architecture).fit(self.maps, self.y)
                probabilities = model.predict_proba(self.maps)
                np.testing.assert_array_equal(model.classes_, [4, 11, 27])
                self.assertEqual(probabilities.shape, (len(self.y), 3))
                self.assertTrue(np.all(np.isfinite(probabilities)))
                self.assertTrue(np.all(probabilities >= 0))
                np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-12)
                np.testing.assert_array_equal(
                    model.predict(self.maps), model.classes_[np.argmax(probabilities, axis=1)],
                )

    def test_late_mean_matches_independent_readout_softmax_calculation(self):
        temperature = 0.65
        model = make_model("parallel_late", temperature=temperature).fit(self.maps, self.y)
        features = model.extract_features(self.maps)
        states = np.split(features, np.cumsum(model.node_counts_)[:-1], axis=1)
        probabilities = [softmax(branch @ readout / temperature)
                         for branch, readout in zip(states, model.readouts_)]
        np.testing.assert_allclose(model.predict_proba(self.maps), np.mean(probabilities, axis=0))

    def test_model_construction_and_fit_do_not_modify_global_numpy_rng(self):
        saved_state = np.random.get_state()
        try:
            np.random.seed(173)
            before = np.random.get_state()
            first = make_model("parallel_intermediate").fit(self.maps, self.y)
            after = np.random.get_state()
            self.assertEqual(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(before[2:], after[2:])
            np.random.seed(912)
            make_model("single_early", random_state=23).fit(self.maps, self.y)
            second = make_model("parallel_intermediate").fit(self.maps, self.y)
            np.testing.assert_array_equal(first.extract_features(self.maps), second.extract_features(self.maps))
            np.testing.assert_array_equal(first.predict_proba(self.maps), second.predict_proba(self.maps))
        finally:
            np.random.set_state(saved_state)

    def test_scalers_use_only_training_values(self):
        model = make_model("parallel_intermediate").fit(self.maps, self.y)
        for name, sequences in self.maps.items():
            values = np.concatenate(sequences, axis=0)
            np.testing.assert_allclose(model.scalers_[name]["mean"], values.mean(axis=0))
            np.testing.assert_allclose(model.scalers_[name]["scale"], values.std(axis=0))

    def test_predictions_do_not_depend_on_test_batch_scaling(self):
        one = {name: [sequences[0]] for name, sequences in self.maps.items()}
        batch = {name: [sequences[0], sequences[1] + 1e9]
                 for name, sequences in self.maps.items()}
        for architecture in ARCHITECTURES:
            with self.subTest(architecture=architecture):
                model = make_model(architecture).fit(self.maps, self.y)
                original_scalers = {
                    name: {key: value.copy() for key, value in scaler.items()}
                    for name, scaler in model.scalers_.items()
                }
                np.testing.assert_allclose(
                    model.predict_proba(one)[0], model.predict_proba(batch)[0],
                    rtol=1e-14, atol=1e-14,
                )
                for name, scaler in original_scalers.items():
                    for key, value in scaler.items():
                        np.testing.assert_array_equal(model.scalers_[name][key], value)

    def test_invalid_total_budget_and_unknown_architecture_are_rejected(self):
        for architecture in ARCHITECTURES[2:]:
            with self.subTest(architecture=architecture):
                with self.assertRaises(ValueError):
                    make_model(architecture, total_nodes=13).fit(self.maps, self.y)
        for parameters in (dict(total_nodes=0), dict(regularization=-1), dict(temperature=0)):
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                make_model("single_early", **parameters).fit(self.maps, self.y)
        with self.assertRaises(ValueError):
            make_model("unknown_architecture").fit(self.maps, self.y)

    def test_misaligned_and_invalid_sequences_are_rejected(self):
        invalid_maps = [({}, self.y)]
        shorter = dict(self.maps)
        shorter["rtm"] = shorter["rtm"][:-1]
        invalid_maps.append((shorter, self.y))
        invalid_maps.append((self.maps, self.y[:-1]))
        for replacement in (
            np.ones(3), np.empty((0, 3)), np.full((3, 3), np.nan),
            np.ones((self.maps["rtm"][0].shape[0], 4)),
            np.ones((self.maps["rtm"][0].shape[0] + 1, 3)),
        ):
            changed = dict(self.maps)
            changed["rtm"] = list(self.maps["rtm"])
            changed["rtm"][0] = replacement
            invalid_maps.append((changed, self.y))
        for index, (maps, labels) in enumerate(invalid_maps):
            with self.subTest(case=index), self.assertRaises(ValueError):
                make_model("parallel_intermediate").fit(maps, labels)

    def test_prediction_rejects_missing_maps_or_changed_feature_width(self):
        model = make_model("parallel_intermediate").fit(self.maps, self.y)
        missing = {name: data for name, data in self.maps.items() if name != "rtm"}
        changed = dict(self.maps)
        changed["rtm"] = [np.ones((sequence.shape[0], 4)) for sequence in self.maps["rtm"]]
        for maps in [missing, changed]:
            with self.subTest(map_names=list(maps)), self.assertRaises(ValueError):
                model.predict_proba(maps)


class ProbabilityFusionTests(unittest.TestCase):
    def setUp(self):
        self.probabilities = np.array([
            [[0.6, 0.3, 0.1], [0.1, 0.2, 0.7]],
            [[0.2, 0.3, 0.5], [0.6, 0.3, 0.1]],
            [[0.3, 0.4, 0.3], [0.2, 0.7, 0.1]],
        ])

    def test_equal_weight_fusion_rules_match_independent_formulas(self):
        formulas = {
            "mean": self.probabilities.mean(axis=0),
            "geometric": np.exp(np.log(self.probabilities).mean(axis=0)),
            "product": self.probabilities.prod(axis=0),
            "max": self.probabilities.max(axis=0),
        }
        for method, expected in formulas.items():
            with self.subTest(method=method):
                expected /= expected.sum(axis=1, keepdims=True)
                np.testing.assert_allclose(fuse_probabilities(self.probabilities, method=method), expected)

    def test_weighted_mean_and_geometric_accept_relative_weights(self):
        weights = np.array([1, 2, 5], dtype=float)
        normalized = weights / weights.sum()
        formulas = {
            "mean": np.sum(self.probabilities * normalized[:, None, None], axis=0),
            "geometric": np.exp(np.sum(np.log(self.probabilities) * normalized[:, None, None], axis=0)),
        }
        for method, expected in formulas.items():
            with self.subTest(method=method):
                expected /= expected.sum(axis=1, keepdims=True)
                np.testing.assert_allclose(
                    fuse_probabilities(self.probabilities, method=method, weights=weights), expected,
                )

    def test_extreme_probabilities_stay_finite_and_normalized(self):
        probabilities = np.array([[[1, 0, 0]], [[0, 1, 0]], [[1e-250, 1e-250, 1]]], dtype=float)
        for method in ("mean", "geometric", "product", "max"):
            with self.subTest(method=method):
                result = fuse_probabilities(probabilities, method=method)
                self.assertTrue(np.isfinite(result).all())
                self.assertTrue((result >= 0).all())
                np.testing.assert_allclose(result.sum(axis=1), 1, atol=1e-12)

    def test_invalid_fusion_rules_and_weights_are_rejected(self):
        with self.assertRaises(ValueError):
            fuse_probabilities(self.probabilities, method="unknown_rule")
        for weights in ([1, 2], [0, 0, 0], [1, -1, 1], [1, np.nan, 1]):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                fuse_probabilities(self.probabilities, weights=weights)


class FusionEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.maps, self.y = synthetic_maps()
        self.metadata = [dict(subject=i // 9, session=(i // 3) % 3, gesture=int(label))
                         for i, label in enumerate(self.y)]
        self.config = dict(
            total_nodes=12, regularization=0.17,
            reservoir_params={name: value for name, value in PARAMETERS.items()
                              if name not in ("total_nodes", "regularization", "random_state")},
            seeds=(19,), protocols=("50_50",), split_seed=41, progress=False,
        )

    def test_soli_maps_preserve_channel_order_and_do_not_copy_data(self):
        md = {1: self.maps["md"], 0: self.maps["angle"]}
        rtm = {1: self.maps["rtm"], 0: self.maps["angle"]}
        maps = build_soli_maps(md, rtm, channels=(0, 1))
        self.assertEqual(list(maps), ["DTM_ch0", "RTM_ch0", "DTM_ch1", "RTM_ch1"])
        self.assertIs(maps["DTM_ch1"], md[1])
        self.assertIs(maps["RTM_ch0"], rtm[0])
        with self.assertRaises(ValueError):
            build_soli_maps(md, {0: rtm[0]})

    def test_outer_splits_are_disjoint_and_preserve_requested_groups(self):
        splits = make_soli_splits(self.y, self.metadata, n_splits=3, split_seed=41)
        all_indices = set(range(len(self.y)))
        for split in splits:
            with self.subTest(protocol=split.protocol, fold=split.fold):
                train, test = set(split.train_indices), set(split.test_indices)
                self.assertTrue(train)
                self.assertTrue(test)
                self.assertFalse(train & test)
                if split.protocol != "session_split":
                    self.assertEqual(train | test, all_indices)
                if split.protocol == "loso":
                    self.assertFalse(
                        {self.metadata[i]["subject"] for i in train} &
                        {self.metadata[i]["subject"] for i in test},
                    )
                if split.protocol == "session_split":
                    self.assertEqual(len({self.metadata[i]["subject"] for i in train | test}), 1)
                    self.assertFalse(
                        {self.metadata[i]["session"] for i in train} &
                        {self.metadata[i]["session"] for i in test},
                    )
        half = [split for split in splits if split.protocol == "50_50"]
        np.testing.assert_array_equal(half[0].train_indices, half[1].test_indices)
        np.testing.assert_array_equal(half[0].test_indices, half[1].train_indices)
        for protocol in ("10fold", "loso"):
            held_out = np.concatenate([split.test_indices for split in splits if split.protocol == protocol])
            np.testing.assert_array_equal(np.sort(held_out), np.arange(len(self.y)))

    def test_fixed_comparison_uses_identical_budget_seeds_and_readout_parameters(self):
        results = run_fusion_comparison(self.maps, self.y, self.metadata, **self.config)
        expected_methods = {
            "Single-map/md", "Single-map/rtm", "Single-map/angle", "Single-Early",
            "Parallel-Early", "Parallel-Intermediate", "Parallel-Late/mean",
            "Parallel-Late/geometric", "Parallel-Late/product", "Parallel-Late/max",
        }
        self.assertEqual(results["search"], [])
        for split in results["splits"]:
            rows = [row for row in results["records"]
                    if row["protocol"] == split["protocol"] and row["fold"] == split["fold"]]
            self.assertEqual({row["method"] for row in rows}, expected_methods)
            self.assertEqual(len(rows), len(expected_methods))
            for row in rows:
                self.assertEqual(row["seed"], 19)
                self.assertEqual(row["total_nodes"], 12)
                self.assertEqual(sum(row["nodes_per_reservoir"]), 12)
                self.assertEqual(row["regularization"], 0.17)
                self.assertEqual(row["n_search_trials"], 0)
                self.assertEqual(row["n_train"], len(split["train_indices"]))
                self.assertEqual(row["n_test"], len(split["test_indices"]))
        self.assertEqual(len(results["contrasts"]), 3)
        self.assertTrue(all(row["n_pairs"] == 2 for row in results["contrasts"]))

    def test_nested_search_excludes_outer_test_and_gives_every_method_same_trials(self):
        captured_fits = []
        identities = {id(sequence): i for i, sequence in enumerate(self.maps["md"])}
        original_fit = FusionESN.fit

        def recording_fit(model, maps, labels):
            indices = [identities[id(sequence)] for sequence in maps["md"]]
            np.testing.assert_array_equal(labels, self.y[indices])
            captured_fits.append(dict(indices=indices, architecture=model.architecture,
                                      regularization=model.regularization))
            return original_fit(model, maps, labels)

        with patch.object(FusionESN, "fit", recording_fit):
            results = run_fusion_comparison(
                self.maps, self.y, self.metadata, **self.config, n_trials=2,
                parameter_candidates=[{"regularization": 0.01}, {"regularization": 0.3}],
            )
        models_per_evaluation = len(self.maps) + 4
        fits_per_outer = models_per_evaluation * 3  # Two search candidates and one final fit.
        self.assertEqual(len(captured_fits), fits_per_outer * len(results["splits"]))
        self.assertEqual(len(results["search"]), len(results["splits"]))
        for fold_index, (outer, search) in enumerate(zip(results["splits"], results["search"])):
            outer_train, outer_test = set(outer["train_indices"]), set(outer["test_indices"])
            inner_train, validation = set(search["train_indices"]), set(search["validation_indices"])
            self.assertFalse(inner_train & validation)
            self.assertEqual(inner_train | validation, outer_train)
            self.assertFalse((inner_train | validation) & outer_test)
            self.assertEqual(len(search["trials"]), 2)
            fits = captured_fits[fold_index * fits_per_outer:(fold_index + 1) * fits_per_outer]
            for fit in fits[:models_per_evaluation * 2]:
                self.assertEqual(set(fit["indices"]), inner_train)
            for fit in fits[models_per_evaluation * 2:]:
                self.assertEqual(set(fit["indices"]), outer_train)
            selected = search["selected_candidate"]
            selected_lambda = search["trials"][selected]["parameters"]["regularization"]
            rows = [row for row in results["records"] if row["fold"] == outer["fold"]]
            self.assertTrue(all(row["selected_candidate"] == selected for row in rows))
            self.assertTrue(all(row["regularization"] == selected_lambda for row in rows))
            self.assertTrue(all(row["n_search_trials"] == 2 for row in rows))
            for trial in search["trials"]:
                accuracies = trial["method_accuracies"]
                baseline_mean = np.mean([score for method, score in accuracies.items()
                                         if method.startswith("Single-map/")])
                other_scores = [score for method, score in accuracies.items()
                                if not method.startswith("Single-map/")]
                self.assertEqual(len(other_scores), 4)
                self.assertAlmostEqual(trial["family_mean_accuracy"], np.mean([baseline_mean, *other_scores]))

    def test_nested_group_validation_keeps_subjects_separate(self):
        # Four subjects leave three training subject groups in every LOSO fold.
        maps = {name: sequences * 2 for name, sequences in self.maps.items()}
        labels = np.tile(self.y, 2)
        metadata = [dict(subject=i // 9, session=(i // 3) % 3, gesture=int(label))
                    for i, label in enumerate(labels)]
        config = dict(self.config, protocols=("loso",))
        results = run_fusion_comparison(
            maps, labels, metadata, **config, n_trials=2,
            parameter_candidates=[{"regularization": 0.01}, {"regularization": 0.3}],
        )
        for outer, search in zip(results["splits"], results["search"]):
            outer_test_subjects = {metadata[i]["subject"] for i in outer["test_indices"]}
            train_subjects = {metadata[i]["subject"] for i in search["train_indices"]}
            validation_subjects = {metadata[i]["subject"] for i in search["validation_indices"]}
            self.assertFalse(train_subjects & validation_subjects)
            self.assertFalse(outer_test_subjects & (train_subjects | validation_subjects))

    def test_inner_holdout_is_fixed_across_reservoir_seeds_and_final_parameters_are_shared(self):
        config = dict(self.config, seeds=(19, 20))
        result = run_fusion_comparison(
            self.maps, self.y, self.metadata, **config, n_trials=2,
            parameter_candidates=[{"regularization": 0.01}, {"regularization": 0.3}],
        )
        self.assertEqual(len(result["search"]), len(result["splits"]) * 2)
        for outer in result["splits"]:
            searches = [search for search in result["search"]
                        if search["protocol"] == outer["protocol"] and search["fold"] == outer["fold"]]
            self.assertEqual({search["seed"] for search in searches}, {19, 20})
            self.assertEqual(searches[0]["train_indices"], searches[1]["train_indices"])
            self.assertEqual(searches[0]["validation_indices"], searches[1]["validation_indices"])
            for search in searches:
                self.assertEqual(search["inner_split_seed"], config["split_seed"])
                selected = search["selected_candidate"]
                expected_lambda = search["trials"][selected]["parameters"]["regularization"]
                rows = [row for row in result["records"]
                        if row["protocol"] == outer["protocol"] and row["fold"] == outer["fold"]
                        and row["seed"] == search["seed"]]
                self.assertEqual(len(rows), 10)
                self.assertTrue(all(row["regularization"] == expected_lambda for row in rows))
                self.assertTrue(all(row["selected_candidate"] == selected for row in rows))
                self.assertTrue(all(row["n_search_trials"] == 2 for row in rows))

    def test_numpy_parameters_and_path_report_metadata_serialize_successfully(self):
        config = dict(self.config, regularization=np.float32(0.17))
        result = run_fusion_comparison(self.maps, self.y, self.metadata, **config)
        for candidate in result["configuration"]["parameter_candidates"]:
            self.assertIsInstance(candidate["regularization"], float)
            self.assertNotIsInstance(candidate["regularization"], np.generic)
        result["configuration"]["data"] = dict(base_dir=Path("/tmp/synthetic-radar"))
        result["configuration"]["report_metadata"] = dict(
            source=Path("reference.h5"), sample_ids=np.array([4, 11, 27]),
            count=np.int64(18), normalized=np.bool_(True), scaling=np.float32(0.2),
        )
        with tempfile.TemporaryDirectory() as temporary:
            save_fusion_results(result, Path(temporary))
            with (Path(temporary) / "fusion_results.json").open(encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual(saved["configuration"]["data"]["base_dir"], "/tmp/synthetic-radar")
            metadata = saved["configuration"]["report_metadata"]
            self.assertEqual(metadata["source"], "reference.h5")
            self.assertEqual(metadata["sample_ids"], [4, 11, 27])
            self.assertEqual(metadata["count"], 18)
            self.assertIs(metadata["normalized"], True)
            self.assertAlmostEqual(metadata["scaling"], 0.2)
            self.assertAlmostEqual(saved["records"][0]["regularization"], float(np.float32(0.17)))

    def test_search_candidate_count_and_forbidden_node_changes_are_rejected(self):
        cases = [
            dict(n_trials=2),
            dict(n_trials=2, parameter_candidates=[{"regularization": 0.1}]),
            dict(parameter_candidates=[{"total_nodes": 24}]),
            dict(parameter_candidates=[{"random_state": 999}]),
        ]
        for parameters in cases:
            with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                run_fusion_comparison(self.maps, self.y, self.metadata, **self.config, **parameters)

    def test_run_all_adds_fusion_once_and_reuses_the_loaded_dataset(self):
        entry = load_soli_entrypoint("run_all.py")
        md, rtm = {0: self.maps["md"]}, {0: self.maps["rtm"]}
        data_config = dict(channels=[0], base_dir="/unused", max_samples_per_gesture_subject=3)
        fusion_config = dict(total_nodes=8, regularization=0.05, seeds=[19])
        legacy_result = {
            "50_50": dict(accuracy_pattern1=0.5, accuracy_pattern2=0.5),
            **{protocol: dict(mean_accuracy=0.5, std_accuracy=0.0)
               for protocol in ("10fold", "session_split", "loso")},
        }
        fusion_result = {"records": [{"method": "Single-Early"}]}
        with patch.object(entry, "DualDataTypeLoader") as loader, \
                patch.object(entry, "run_soli_evaluation", return_value=legacy_result) as legacy, \
                patch.object(entry, "run_fusion_comparison", return_value=fusion_result) as fusion, \
                contextlib.redirect_stdout(io.StringIO()):
            loader.return_value.load_gesture_data.return_value = md, rtm, self.y, self.metadata
            result = entry.main(data_config=data_config, fusion_config=fusion_config)
        loader.assert_called_once_with(channels=[0], base_dir="/unused")
        loader.return_value.load_gesture_data.assert_called_once_with(max_samples_per_gesture_subject=3)
        self.assertEqual(legacy.call_count, 7)
        fusion.assert_called_once()
        arguments = fusion.call_args.kwargs
        self.assertIs(arguments["data_config"], data_config)
        self.assertIs(arguments["fusion_config"], fusion_config)
        for actual, expected in zip(arguments["loaded_data"], (md, rtm, self.y, self.metadata)):
            self.assertIs(actual, expected)
        self.assertEqual(len(result), 8)
        self.assertIs(result["fusion_comparison"], fusion_result)
        self.assertEqual(list(result)[:-1], ["Multi_RR_L", "Multi_RR_N", "Multi_SVM", "Multi_RF",
                                           "Single_RF", "Single_SVM", "Single_Ridge"])

    def test_quick_cli_forwards_reproducible_defaults_and_explicit_overrides(self):
        entry = load_soli_entrypoint("run_fusion.py")
        with tempfile.TemporaryDirectory() as temporary, patch.object(entry, "main") as main:
            entry.cli(["--quick", "--output-dir", temporary])
            data, config = main.call_args.args
            self.assertEqual(data["max_samples_per_gesture_subject"], 2)
            self.assertEqual(config["total_nodes"], 32)
            self.assertEqual(config["seeds"], [42])
            self.assertEqual(config["protocols"], ["50_50"])
            self.assertEqual(config["n_trials"], 1)
            self.assertIsNone(config["parameter_candidates"])
            self.assertEqual(config["output_dir"], str(Path(temporary).resolve()))
            main.reset_mock()
            entry.cli(["--quick", "--total-nodes", "16", "--regularization", "0.03",
                       "--seeds", "5,7", "--protocols", "50_50,loso", "--max-samples", "4"])
            data, config = main.call_args.args
            self.assertEqual(data["max_samples_per_gesture_subject"], 4)
            self.assertEqual(config["total_nodes"], 16)
            self.assertEqual(config["regularization"], 0.03)
            self.assertEqual(config["seeds"], [5, 7])
            self.assertEqual(config["protocols"], ["50_50", "loso"])

    def test_fusion_entrypoint_saves_complete_reports_without_reloading_data(self):
        entry = load_soli_entrypoint("run_fusion.py")
        md = {0: self.maps["md"], 1: self.maps["angle"]}
        rtm = {0: self.maps["rtm"], 1: self.maps["md"]}
        metadata = [dict(row, filename=f"sample_{i}.h5") for i, row in enumerate(self.metadata)]
        data_config = dict(channels=[0, 1], base_dir="/unused", max_samples_per_gesture_subject=3)
        with tempfile.TemporaryDirectory() as temporary:
            config = dict(self.config, total_nodes=8, output_dir=temporary)
            original_config = config.copy()
            with patch.object(entry, "DualDataTypeLoader") as loader, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = entry.main(data_config, config, loaded_data=(md, rtm, self.y, metadata))
            loader.assert_not_called()
            self.assertEqual(config, original_config)
            directory = Path(temporary)
            self.assertEqual({path.name for path in directory.iterdir() if path.is_file()}, {
                "fusion_results.json", "fusion_summary.csv", "fusion_records.csv", "fusion_contrasts.csv",
                "fusion_summary.png",
            })
            self.assertEqual((directory / "fusion_summary.png").read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            with (directory / "fusion_results.json").open(encoding="utf-8") as handle:
                saved = json.load(handle)
            self.assertEqual(saved["configuration"]["data"], data_config)
            self.assertEqual(saved["configuration"]["sample_filenames"], [row["filename"] for row in metadata])
            self.assertEqual(saved["configuration"]["map_names"], ["DTM_ch0", "RTM_ch0", "DTM_ch1", "RTM_ch1"])
            self.assertEqual(len(saved["records"]), 22)
            self.assertEqual(saved["records"], result["records"])
            for filename, key in (("fusion_summary.csv", "summary"),
                                  ("fusion_records.csv", "records"),
                                  ("fusion_contrasts.csv", "contrasts")):
                with (directory / filename).open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), len(saved[key]))
                self.assertTrue(rows)


if __name__ == "__main__":
    unittest.main()
