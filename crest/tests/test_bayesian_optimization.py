"""Bayesian search budgets, independent method tuning, and nested data isolation."""

import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.bayesian_optimization import bayesian_optimize
from modules.fusion import FusionESN
from modules.fusion_evaluation import run_fusion_comparison, save_fusion_results


BASE_PARAMETERS = dict(
    spectral_radius=0.95, input_scaling=0.2, density=0.1,
    leakage_rate=0.05, temperature=1.0, standardize_inputs=True,
    regularization=0.17, bias_scaling=0.0,
)
SEARCH_KEYS = {
    "spectral_radius", "input_scaling", "density", "leakage_rate",
    "temperature", "standardize_inputs",
}


def synthetic_maps():
    rng = np.random.default_rng(701)
    labels = np.tile([4, 11], 12)
    maps = {"md": [], "rtm": []}
    for sample, label in enumerate(labels):
        for name, width in (("md", 2), ("rtm", 3)):
            maps[name].append(rng.normal(label / 10, 0.7, (3 + sample % 3, width)))
    metadata = [dict(subject=i // 6, session=(i // 2) % 3, gesture=int(label))
                for i, label in enumerate(labels)]
    return maps, labels, metadata


def comparison_config(**overrides):
    return dict(total_nodes=8, regularization=0.17, seeds=(19,), protocols=("50_50",),
                split_seed=41, search_strategy="bayesian", progress=False, **overrides)


class BayesianOptimizerTests(unittest.TestCase):
    def test_fixed_budget_includes_baseline_and_keeps_unsearched_parameters(self):
        evaluated = []

        def objective(parameters):
            evaluated.append(dict(parameters))
            return dict(score=-abs(parameters["input_scaling"] - 0.4),
                        accuracy=0.5, log_loss=0.7)

        result = bayesian_optimize(objective, base_parameters=BASE_PARAMETERS,
                                   n_trials=7, n_initial_points=3, random_state=12)
        self.assertEqual(result["n_trials"], 7)
        self.assertEqual(len(result["trials"]), 7)
        self.assertEqual(len(evaluated), 7)
        self.assertEqual(result["trials"][0]["parameters"], BASE_PARAMETERS)
        self.assertEqual(result["n_initial_points"], 3)
        self.assertEqual([trial["candidate"] for trial in result["trials"]], list(range(7)))
        self.assertEqual([trial["phase"] for trial in result["trials"]],
                         ["initial"] * 3 + ["bayesian"] * 4)
        self.assertEqual(set(result["search_space"]), SEARCH_KEYS)
        for trial, parameters in zip(result["trials"], evaluated):
            self.assertEqual(trial["parameters"], parameters)
            self.assertEqual(parameters["regularization"], 0.17)
            self.assertEqual(parameters["bias_scaling"], 0.0)
            self.assertIsInstance(parameters["standardize_inputs"], bool)
            self.assertTrue(np.isfinite(trial["score"]))
        selected = result["selected_candidate"]
        self.assertEqual(result["parameters"], evaluated[selected])
        self.assertEqual(result["trials"][selected]["score"],
                         max(trial["score"] for trial in result["trials"]))
        json.dumps(result, allow_nan=False)

    def test_small_budget_is_exact_and_never_expands_to_initial_design_size(self):
        for count in (1, 2, 4):
            with self.subTest(n_trials=count):
                calls = []

                def objective(parameters):
                    calls.append(parameters)
                    return {"score": 0.5}

                result = bayesian_optimize(objective, base_parameters=BASE_PARAMETERS,
                                           n_trials=count)
                self.assertEqual(len(calls), count)
                self.assertEqual(len(result["trials"]), count)
                self.assertEqual(result["n_initial_points"], count)

    def test_public_default_budget_is_twelve_initial_and_forty_eight_adaptive_trials(self):
        calls = []

        def objective(parameters):
            calls.append(parameters)
            return {"score": 0.5}

        with patch("modules.bayesian_optimization._propose",
                   return_value=np.full(len(SEARCH_KEYS), 0.5)) as proposal:
            result = bayesian_optimize(objective, base_parameters=BASE_PARAMETERS)
        self.assertEqual(result["n_trials"], 60)
        self.assertEqual(result["n_initial_points"], 12)
        self.assertEqual(len(calls), 60)
        self.assertEqual(proposal.call_count, 48)
        self.assertEqual([trial["phase"] for trial in result["trials"]],
                         ["initial"] * 12 + ["bayesian"] * 48)

    def test_reproducibility_and_local_rng_with_adaptive_trials(self):
        def objective(parameters):
            return {"score": -(np.log(parameters["input_scaling"] / 0.4) ** 2)}

        saved_state = np.random.get_state()
        try:
            np.random.seed(371)
            before = np.random.get_state()
            first = bayesian_optimize(objective, base_parameters=BASE_PARAMETERS,
                                      n_trials=6, n_initial_points=2, random_state=82)
            after = np.random.get_state()
            second = bayesian_optimize(objective, base_parameters=BASE_PARAMETERS,
                                       n_trials=6, n_initial_points=2, random_state=82)
            self.assertEqual(before[0], after[0])
            np.testing.assert_array_equal(before[1], after[1])
            self.assertEqual(before[2:], after[2:])
            self.assertEqual(first, second)
        finally:
            np.random.set_state(saved_state)

    def test_bayesian_proposals_respond_to_observations(self):
        def optimize(sign):
            return bayesian_optimize(
                lambda parameters: {"score": sign * np.log(parameters["input_scaling"])},
                base_parameters=BASE_PARAMETERS, n_trials=6,
                n_initial_points=2, random_state=39,
            )

        increasing, decreasing = optimize(1), optimize(-1)
        first_initial = [trial["parameters"] for trial in increasing["trials"][:2]]
        second_initial = [trial["parameters"] for trial in decreasing["trials"][:2]]
        self.assertEqual(first_initial, second_initial)
        self.assertNotEqual(
            [trial["parameters"] for trial in increasing["trials"][2:]],
            [trial["parameters"] for trial in decreasing["trials"][2:]],
        )

    def test_partial_search_space_override_bounds_and_boolean_choices(self):
        overrides = dict(
            input_scaling={"low": 0.1, "high": 0.5, "scale": "log"},
            density={"low": 0.05, "high": 0.2, "scale": "linear"},
            standardize_inputs={"choices": [True, False]},
        )
        result = bayesian_optimize(lambda parameters: {"score": parameters["density"]},
                                   base_parameters=BASE_PARAMETERS, n_trials=5,
                                   n_initial_points=2, random_state=73, search_space=overrides)
        self.assertEqual(set(result["search_space"]), SEARCH_KEYS)
        for trial in result["trials"]:
            parameters = trial["parameters"]
            self.assertLessEqual(0.1, parameters["input_scaling"])
            self.assertLessEqual(parameters["input_scaling"], 0.5)
            self.assertLessEqual(0.05, parameters["density"])
            self.assertLessEqual(parameters["density"], 0.2)
            self.assertIsInstance(parameters["standardize_inputs"], bool)

    def test_invalid_budget_fixed_dimensions_and_objective_scores_are_rejected(self):
        for count in (0, -1, 2.5, True):
            with self.subTest(n_trials=count), self.assertRaises(ValueError):
                bayesian_optimize(lambda parameters: {"score": 0.5},
                                  base_parameters=BASE_PARAMETERS, n_trials=count)
        for key in ("total_nodes", "regularization", "bias_scaling", "random_state", "unknown"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                bayesian_optimize(lambda parameters: {"score": 0.5},
                                  base_parameters=BASE_PARAMETERS, n_trials=1,
                                  search_space={key: {"low": 0.1, "high": 1.0, "scale": "linear"}})
        for score in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(score=score), self.assertRaises(ValueError):
                bayesian_optimize(lambda parameters, score=score: {"score": score},
                                  base_parameters=BASE_PARAMETERS, n_trials=1)

    def test_zero_density_or_leakage_bounds_are_rejected_before_evaluation(self):
        for key in ("density", "leakage_rate"):
            with self.subTest(key=key), patch("modules.bayesian_optimization._propose") as propose:
                calls = []
                with self.assertRaises(ValueError):
                    bayesian_optimize(lambda parameters: calls.append(parameters),
                                      base_parameters=BASE_PARAMETERS, n_trials=3,
                                      search_space={key: {"low": 0., "high": 1., "scale": "linear"}})
                self.assertEqual(calls, [])
                propose.assert_not_called()


class BayesianFusionComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.maps, cls.y, cls.metadata = synthetic_maps()
        cls.expected_methods = {
            "Single-map/md", "Single-map/rtm", "Single-Early", "Parallel-Early",
            "Parallel-Intermediate", "Parallel-Late/mean", "Parallel-Late/product",
            "Parallel-Late/geometric", "Parallel-Late/max",
        }
        identities = {id(sequence): index for index, sequence in enumerate(cls.maps["md"])}
        original_fit = FusionESN.fit
        cls.fits = []

        def recording_fit(model, maps, labels):
            indices = [identities[id(sequence)] for sequence in maps["md"]]
            np.testing.assert_array_equal(labels, cls.y[indices])
            result = original_fit(model, maps, labels)
            cls.fits.append(dict(indices=set(indices), architecture=model.architecture,
                                 total_nodes=sum(model.node_counts_),
                                 nodes=list(model.node_counts_), regularization=model.regularization,
                                 bias_scaling=model.bias_scaling))
            return result

        with patch.object(FusionESN, "fit", recording_fit):
            cls.results = run_fusion_comparison(cls.maps, cls.y, cls.metadata,
                                               **comparison_config(n_trials=4))

    def test_every_reported_method_gets_exact_budget_and_its_own_selection(self):
        result = self.results
        self.assertEqual(result["configuration"]["n_trials"], 4)
        self.assertEqual(len(result["search"]), len(result["splits"]) * len(self.expected_methods))
        searches = {(search["protocol"], search["fold"], search["seed"], search["method"]): search
                    for search in result["search"]}
        for outer in result["splits"]:
            rows = [row for row in result["records"] if row["fold"] == outer["fold"]]
            self.assertEqual({row["method"] for row in rows}, self.expected_methods)
            for row in rows:
                search = searches[row["protocol"], row["fold"], row["seed"], row["method"]]
                self.assertEqual(search["n_trials"], 4)
                self.assertEqual(len(search["trials"]), 4)
                self.assertEqual(row["n_search_trials"], 4)
                self.assertEqual(row["selected_candidate"], search["selected_candidate"])
                best = search["trials"][search["selected_candidate"]]
                expected = max(search["trials"], key=lambda trial: (trial["accuracy"], -trial["log_loss"]))
                self.assertEqual(best, expected)
                self.assertEqual(row["parameters"], best["parameters"])
                self.assertEqual(row["total_nodes"], 8)
                self.assertEqual(row["regularization"], 0.17)
                self.assertEqual(row["parameters"]["bias_scaling"], 0.0)
                self.assertEqual(sum(row["nodes_per_reservoir"]), 8)
                expected_nodes = [4, 4] if row["architecture"].startswith("parallel") else [8]
                self.assertEqual(row["nodes_per_reservoir"], expected_nodes)
                for trial in search["trials"]:
                    self.assertTrue({"accuracy", "log_loss", "score", "phase"} <= set(trial))
                    self.assertEqual(trial["parameters"]["regularization"], 0.17)
                    self.assertEqual(trial["parameters"]["bias_scaling"], 0.0)

    def test_inner_search_fits_only_training_data_and_holdout_is_shared(self):
        expected_fit_count = len(self.results["splits"]) * len(self.expected_methods) * (4 + 1)
        self.assertEqual(len(self.fits), expected_fit_count)
        for outer in self.results["splits"]:
            searches = [search for search in self.results["search"] if search["fold"] == outer["fold"]]
            inner = set(searches[0]["train_indices"])
            validation = set(searches[0]["validation_indices"])
            outer_train, outer_test = set(outer["train_indices"]), set(outer["test_indices"])
            self.assertFalse(inner & validation)
            self.assertEqual(inner | validation, outer_train)
            self.assertFalse((inner | validation) & outer_test)
            self.assertTrue(all(set(search["train_indices"]) == inner for search in searches))
            self.assertTrue(all(set(search["validation_indices"]) == validation for search in searches))
            self.assertEqual(sum(fit["indices"] == inner for fit in self.fits),
                             4 * len(self.expected_methods))
            self.assertEqual(sum(fit["indices"] == outer_train for fit in self.fits),
                             len(self.expected_methods))
        for fit in self.fits:
            self.assertEqual(fit["total_nodes"], 8)
            self.assertEqual(fit["regularization"], 0.17)
            self.assertEqual(fit["bias_scaling"], 0.0)

    def test_default_budget_and_method_specific_parameters_are_forwarded(self):
        selected_scalings = []

        def controlled_optimizer(objective, *, base_parameters, n_trials=60,
                                 random_state=42, search_space=None, n_initial_points=12):
            scaling = 0.25 + len(selected_scalings) * 0.05
            selected_scalings.append(scaling)
            selected = dict(base_parameters, input_scaling=scaling)
            trials = [dict(candidate=index,
                           parameters=dict(base_parameters) if index < n_trials - 1 else selected,
                           phase="initial" if index < n_initial_points else "bayesian",
                           score=float(index), accuracy=0.5, log_loss=0.7)
                      for index in range(n_trials)]
            return dict(parameters=selected, selected_candidate=n_trials - 1, trials=trials,
                        n_trials=n_trials, n_initial_points=min(n_initial_points, n_trials),
                        search_space={} if search_space is None else search_space)

        with patch("modules.fusion_evaluation.bayesian_optimize", side_effect=controlled_optimizer) as optimizer:
            result = run_fusion_comparison(self.maps, self.y, self.metadata, **comparison_config())
        self.assertEqual(optimizer.call_count, len(result["splits"]) * len(self.expected_methods))
        self.assertEqual(result["configuration"]["n_trials"], 60)
        self.assertEqual({row["n_search_trials"] for row in result["records"]}, {60})
        for outer in result["splits"]:
            rows = [row for row in result["records"] if row["fold"] == outer["fold"]]
            self.assertEqual(len({row["parameters"]["input_scaling"] for row in rows}), len(rows))
            searches = {search["method"]: search for search in result["search"]
                        if search["fold"] == outer["fold"]}
            for row in rows:
                trial = searches[row["method"]]["trials"][row["selected_candidate"]]
                self.assertEqual(row["parameters"], trial["parameters"])

    def test_search_history_and_selected_parameters_persist_in_json_and_csv(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = save_fusion_results(self.results, temporary)
            saved = json.loads((directory / "fusion_results.json").read_text(encoding="utf-8"))
            self.assertEqual(saved["search"], self.results["search"])
            self.assertEqual(saved["records"], self.results["records"])
            with (directory / "fusion_records.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), len(saved["records"]))
            for row, expected in zip(rows, saved["records"]):
                self.assertEqual(json.loads(row["parameters"]), expected["parameters"])
                self.assertEqual(int(row["n_search_trials"]), 4)
                self.assertEqual(int(row["selected_candidate"]), expected["selected_candidate"])

    def test_group_holdout_is_shared_across_methods_and_reservoir_seeds(self):
        config = comparison_config(n_trials=1)
        config.update(protocols=("loso",), seeds=(19, 20), late_fusion_methods=("mean",))
        result = run_fusion_comparison(self.maps, self.y, self.metadata, **config)
        for outer in result["splits"]:
            searches = [search for search in result["search"] if search["fold"] == outer["fold"]]
            self.assertEqual({search["seed"] for search in searches}, {19, 20})
            first = searches[0]
            self.assertTrue(all(search["train_indices"] == first["train_indices"] for search in searches))
            self.assertTrue(all(search["validation_indices"] == first["validation_indices"]
                                for search in searches))
            train_subjects = {self.metadata[index]["subject"] for index in first["train_indices"]}
            validation_subjects = {self.metadata[index]["subject"]
                                   for index in first["validation_indices"]}
            test_subjects = {self.metadata[index]["subject"] for index in outer["test_indices"]}
            self.assertFalse(train_subjects & validation_subjects)
            self.assertFalse((train_subjects | validation_subjects) & test_subjects)
            self.assertTrue(all(search["n_trials"] == 1 for search in searches))

    def test_fixed_dimensions_bias_and_shared_candidates_cannot_enter_bayesian_search(self):
        invalid = [
            dict(parameter_candidates=[{}] * 4),
            dict(reservoir_params={"bias_scaling": 0.1}),
        ] + [dict(bayesian_search_space={key: {"low": 0.01, "high": 1.0, "scale": "linear"}})
             for key in ("total_nodes", "regularization", "bias_scaling", "random_state")]
        for overrides in invalid:
            with self.subTest(overrides=overrides), self.assertRaises(ValueError):
                run_fusion_comparison(self.maps, self.y, self.metadata,
                                      **comparison_config(n_trials=4, **overrides))

    def test_one_way_evaluates_only_the_first_outer_split_with_the_full_method_budget(self):
        result = run_fusion_comparison(self.maps, self.y, self.metadata,
                                      **comparison_config(n_trials=2, bidirectional_50_50=False))
        self.assertEqual(len(result["splits"]), 1)
        self.assertEqual(result["splits"][0], self.results["splits"][0])
        self.assertFalse(result["configuration"]["bidirectional_50_50"])
        self.assertEqual(len(result["records"]), len(self.expected_methods))
        self.assertEqual(len(result["search"]), len(self.expected_methods))
        self.assertTrue(all(row["fold"] == "pattern1" and row["n_search_trials"] == 2
                            for row in result["records"]))
        self.assertTrue(all(len(search["trials"]) == 2 for search in result["search"]))

    def test_invalid_direction_flags_fail_before_model_fitting(self):
        for flag in (0, 1, "false", None):
            with self.subTest(flag=flag), patch.object(FusionESN, "fit") as fit:
                with self.assertRaisesRegex(ValueError, "bidirectional_50_50 must be boolean"):
                    run_fusion_comparison(self.maps, self.y, self.metadata,
                                          **comparison_config(n_trials=2, bidirectional_50_50=flag))
                fit.assert_not_called()


if __name__ == "__main__":
    unittest.main()
