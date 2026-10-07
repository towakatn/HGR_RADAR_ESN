"""Verify high-precision HAR room selection and radar-axis projection."""

import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.har_data import (
    HARDataLoader, HAR_TENSOR_SHAPE, HIGH_PRECISION_DIR,
    discover_har_files, inspect_har_dataset, read_lfs_pointer,
)


def load_har_runner():
    from modules import har_experiment
    return har_experiment


class HARDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base_dir = Path(self.temporary.name)
        self.data_dir = self.base_dir / HIGH_PRECISION_DIR
        self.data_dir.mkdir()

    def save_tensor(self, name="S1_H1_A1_D1_1.npy", tensor=None, directory=None):
        if tensor is None:
            tensor = np.zeros(HAR_TENSOR_SHAPE, dtype=np.float32)
        destination = (self.data_dir if directory is None else directory) / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.save(destination, tensor, allow_pickle=False)
        return destination

    def save_pointer(self, name="S1_H1_A1_D1_1.npy"):
        destination = self.data_dir / name
        destination.write_text(
            "version https://git-lfs.github.com/spec/v1\n"
            "oid sha256:" + "a" * 64 + "\nsize 4194432\n",
            encoding="ascii",
        )
        return destination

    def test_room_selection_excludes_other_rooms_and_one_bit_folder(self):
        self.save_tensor("S3_H1_A10_D2_20.npy")
        self.save_tensor("S1_H1_A2_D1_1.npy")
        # Unmaterialized files in another room must not prevent this experiment.
        self.save_pointer("S1_H2_A2_D1_1.npy")
        one_bit = self.base_dir / "Human Activity_1bit v2.0_Clipping"
        self.save_tensor("S2_H1_A3_D3_2.npy", directory=one_bit)

        samples = discover_har_files(self.base_dir, room=1)
        self.assertEqual([sample.path.name for sample in samples],
                         ["S1_H1_A2_D1_1.npy", "S3_H1_A10_D2_20.npy"])
        loader = HARDataLoader(self.base_dir, room=1)
        maps, labels, metadata = loader.load_all_data()
        np.testing.assert_array_equal(labels, [2, 10])
        self.assertEqual(metadata, [
            dict(filename="S1_H1_A2_D1_1.npy", subject=1, room=1,
                 action=2, distance=1, repetition=1),
            dict(filename="S3_H1_A10_D2_20.npy", subject=3, room=1,
                 action=10, distance=2, repetition=20),
        ])
        self.assertTrue(all(len(sequences) == 2 for sequences in maps.values()))
        self.assertEqual(loader.data_manifest["room"], 1)
        self.assertEqual(loader.data_manifest["n_samples"], 2)
        self.assertEqual(loader.data_manifest["class_counts"], {"2": 1, "10": 1})

    def test_feature_projection_retains_fractional_amplitudes_and_correct_axes(self):
        tensor = np.zeros(HAR_TENSOR_SHAPE, dtype=np.float64)
        tensor[0, 7, 2, 3] = 2.25
        tensor[0, 7, 5, 3] = 4.5
        tensor[0, 7, 2, 9] = 0.125
        tensor[3, 127, 31, 0] = 37.75
        self.save_tensor(tensor=tensor)
        maps, _, _ = HARDataLoader(self.base_dir, room=1).load_all_data()
        self.assertEqual(list(maps), [name for channel in range(4)
                                     for name in (f"DTM_ch{channel}", f"RTM_ch{channel}")])
        expected_dtm = np.zeros((128, 32), dtype=np.float32)
        expected_dtm[7, 3], expected_dtm[7, 9] = 6.75, 0.125
        expected_rtm = np.zeros((128, 32), dtype=np.float32)
        expected_rtm[7, 2], expected_rtm[7, 5] = 2.375, 4.5
        np.testing.assert_array_equal(maps["DTM_ch0"][0], expected_dtm)
        np.testing.assert_array_equal(maps["RTM_ch0"][0], expected_rtm)
        self.assertEqual(maps["DTM_ch3"][0][127, 0], 37.75)
        self.assertEqual(maps["RTM_ch3"][0][127, 31], 37.75)
        for name in ("DTM_ch1", "RTM_ch1", "DTM_ch2", "RTM_ch2"):
            np.testing.assert_array_equal(maps[name][0], 0)
        for sequences in maps.values():
            self.assertEqual(sequences[0].shape, (128, 32))
            self.assertEqual(sequences[0].dtype, np.float32)

    def test_distance_selection_excludes_same_room_unmaterialized_other_distances(self):
        name = "S2_H1_A3_D1_7.npy"
        self.save_tensor(name)
        for excluded in ("S2_H1_A3_D2_7.npy", "S2_H1_A3_D3_7.npy",
                         "S2_H2_A3_D1_7.npy"):
            self.save_pointer(excluded)
        selected = discover_har_files(self.base_dir, room=1, distance=1)
        self.assertEqual([sample.path.name for sample in selected], [name])
        loader = HARDataLoader(self.base_dir, room=1, distance=1)
        maps, labels, metadata = loader.load_all_data()
        np.testing.assert_array_equal(labels, [3])
        self.assertEqual([(row["room"], row["distance"]) for row in metadata], [(1, 1)])
        self.assertTrue(all(len(sequences) == 1 for sequences in maps.values()))
        self.assertEqual(loader.data_manifest["distance"], 1)
        self.assertEqual(loader.data_manifest["distance_meters"], 1.5)
        self.assertEqual(loader.data_manifest["distances"], [1])

    def test_magnitude_conversion_avoids_signed_integer_overflow(self):
        tensor = np.zeros(HAR_TENSOR_SHAPE, dtype=np.int16)
        tensor[2, 6, 11, 4] = -32768
        self.save_tensor(tensor=tensor)
        loader = HARDataLoader(self.base_dir, room=1, channels=(2,))
        maps, _, _ = loader.load_all_data()
        self.assertEqual(maps["DTM_ch2"][0][6, 4], 32768)
        self.assertEqual(maps["RTM_ch2"][0][6, 11], 32768)
        self.assertEqual(loader.data_manifest["source_dtype_counts"], {"int16": 1})

    def test_complex_magnitudes_are_summed_before_projection(self):
        tensor = np.zeros(HAR_TENSOR_SHAPE, dtype=np.complex64)
        tensor[1, 3, 5, 7] = 3 + 4j
        tensor[1, 3, 6, 7] = -3 - 4j
        self.save_tensor(tensor=tensor)
        maps, _, _ = HARDataLoader(self.base_dir, room=1, channels=(1,)).load_all_data()
        self.assertEqual(maps["DTM_ch1"][0][3, 7], 10)
        self.assertEqual(maps["RTM_ch1"][0][3, 5], 5)
        self.assertEqual(maps["RTM_ch1"][0][3, 6], 5)

    def test_missing_lfs_payload_produces_actionable_error(self):
        pointer = self.save_pointer()
        self.assertEqual(read_lfs_pointer(pointer),
                         {"oid": "a" * 64, "size": 4194432})
        loader = HARDataLoader(self.base_dir, room=1)
        with self.assertRaises(FileNotFoundError) as caught:
            loader.load_all_data()
        message = str(caught.exception)
        for fragment in ("H1", "Git LFS", "Download", pointer.name):
            self.assertIn(fragment, message)
        self.assertIsNone(loader.data_manifest)

    def test_inventory_distinguishes_materialized_arrays_and_pointers_by_room(self):
        materialized = self.save_tensor()
        self.save_pointer("S2_H2_A10_D3_20.npy")
        self.assertIsNone(read_lfs_pointer(materialized))
        inventory = inspect_har_dataset(self.base_dir)
        self.assertEqual(inventory["room_counts"], {"1": 1, "2": 1})
        self.assertEqual(inventory["rooms"]["1"]["materialized_npy"], 1)
        self.assertEqual(inventory["rooms"]["2"]["lfs_pointers"], 1)

    def test_loader_rejects_wrong_tensor_shape(self):
        self.save_tensor(tensor=np.zeros((4, 32, 32, 128), dtype=np.float32))
        with self.assertRaisesRegex(ValueError, "shape.*expected"):
            HARDataLoader(self.base_dir, room=1).load_all_data()

    def test_loader_rejects_nonfinite_values_in_any_channel(self):
        for value in (np.nan, np.inf):
            with self.subTest(value=value):
                tensor = np.zeros(HAR_TENSOR_SHAPE, dtype=np.float32)
                tensor[3, 0, 0, 0] = value
                self.save_tensor(tensor=tensor)
                with self.assertRaisesRegex(ValueError, "non-finite"):
                    HARDataLoader(self.base_dir, room=1, channels=(0,)).load_all_data()

    def test_loader_rejects_boolean_and_nonnumeric_arrays(self):
        for dtype in (np.bool_, "U1"):
            with self.subTest(dtype=dtype):
                self.save_tensor(tensor=np.zeros(HAR_TENSOR_SHAPE, dtype=dtype))
                with self.assertRaisesRegex(ValueError, "numeric"):
                    HARDataLoader(self.base_dir, room=1).load_all_data()

    def test_invalid_names_and_duplicate_recording_ids_are_not_silently_skipped(self):
        invalid = self.save_tensor("S1_H1_A11_D1_1.npy")
        with self.assertRaisesRegex(ValueError, "Invalid HAR"):
            discover_har_files(self.base_dir, room=1)
        invalid.unlink()
        name = "S1_H1_A1_D1_1.npy"
        self.save_tensor(name)
        self.save_tensor(name, directory=self.data_dir / "nested")
        with self.assertRaisesRegex(ValueError, "Duplicate HAR"):
            discover_har_files(self.base_dir, room=1)

    def test_explicit_room_and_high_precision_directory_are_required(self):
        for room in (None, True, 0, 5, "1"):
            with self.subTest(room=room), self.assertRaises(ValueError):
                HARDataLoader(self.base_dir, room=room)
        other_dir = self.base_dir / "without_high_precision"
        (other_dir / "Human Activity_1bit v2.0_Clipping").mkdir(parents=True)
        with self.assertRaisesRegex(FileNotFoundError, "16-bit"):
            HARDataLoader(other_dir, room=1)
        with self.assertRaisesRegex(ValueError, "No 16-bit.*H1"):
            HARDataLoader(self.base_dir, room=1).load_all_data()

    def test_loaded_room_runs_paired_bias_free_existing_ridge_comparison(self):
        for action in (1, 10):
            for repetition in range(1, 5):
                tensor = np.zeros(HAR_TENSOR_SHAPE, dtype=np.float32)
                tensor[:, :, action, repetition] = action + repetition / 8
                self.save_tensor(f"S1_H1_A{action}_D1_{repetition}.npy", tensor)
        self.save_pointer("S1_H2_A1_D1_1.npy")
        maps, labels, metadata = HARDataLoader(self.base_dir, room=1, distance=1).load_all_data()
        runner = load_har_runner()
        destination = self.base_dir / "report"
        with contextlib.redirect_stdout(io.StringIO()), patch.object(runner, "save_fusion_plot"):
            results = runner.main(
                data_config={"base_dir": str(self.base_dir), "room": 1, "distance": 1},
                fusion_config=dict(total_nodes=8, seeds=(42,), protocols=("50_50",),
                                   late_fusion_methods=("mean",), n_trials=1,
                                   progress=False, output_dir=str(destination)),
                loaded_data=(maps, labels, metadata),
            )
        self.assertEqual(results["configuration"]["classes"], [1, 10])
        self.assertEqual(results["configuration"]["readout"], "existing_RR_L")
        self.assertEqual(len(results["records"]), 12)
        self.assertEqual(len(results["splits"]), 1)
        self.assertFalse(results["configuration"]["bidirectional_50_50"])
        for split in results["splits"]:
            train, test = set(split["train_indices"]), set(split["test_indices"])
            self.assertFalse(train & test)
            self.assertEqual(train | test, set(range(8)))
            self.assertEqual({metadata[i]["room"] for i in train | test}, {1})
            self.assertEqual({metadata[i]["distance"] for i in train | test}, {1})
            self.assertEqual(sorted(labels[list(train)]), [1, 1, 10, 10])
            self.assertEqual(sorted(labels[list(test)]), [1, 1, 10, 10])
        for row in results["records"]:
            self.assertEqual((row["n_train"], row["n_test"]), (4, 4))
            self.assertEqual(row["total_nodes"], 8)
            self.assertEqual(row["regularization"], 0.1)
            self.assertEqual(row["parameter_counts"]["reservoir_biases"], 0)
            self.assertEqual(row["n_search_trials"], 0)
        with (destination / "split_manifest.csv").open(newline="") as stream:
            membership = list(csv.DictReader(stream))
        self.assertEqual(len(membership), 8)
        self.assertEqual({row["room"] for row in membership}, {"1"})
        self.assertEqual({row["distance"] for row in membership}, {"1"})
        self.assertEqual({row["repeat"] for row in membership}, {"1", "2", "3", "4"})
        persisted = json.loads((destination / "fusion_results.json").read_text())
        self.assertEqual(persisted["configuration"]["session_identifier"], None)
        self.assertEqual(persisted["configuration"]["sample_filenames"],
                         [row["filename"] for row in metadata])
        self.assertEqual(persisted["dataset_summary"]["room"], {"1": 8})

    def test_runner_rejects_mixed_room_loaded_data_before_evaluation(self):
        runner = load_har_runner()
        loaded = ({"DTM_ch0": [np.zeros((128, 32))] * 2}, np.array([1, 10]),
                  [dict(room=1, distance=1), dict(room=2, distance=1)])
        with patch.object(runner, "run_fusion_comparison") as evaluate:
            with self.assertRaisesRegex(ValueError, "single requested room"):
                runner.main(data_config={"room": 1}, loaded_data=loaded)
            evaluate.assert_not_called()

    def test_runner_rejects_mixed_distance_loaded_data_before_evaluation(self):
        runner = load_har_runner()
        loaded = ({"DTM_ch0": [np.zeros((128, 32))] * 2}, np.array([1, 10]),
                  [dict(room=1, distance=1), dict(room=1, distance=2)])
        with patch.object(runner, "run_fusion_comparison") as evaluate:
            with self.assertRaisesRegex(ValueError, "distance"):
                runner.main(data_config={"room": 1, "distance": 1}, loaded_data=loaded)
            evaluate.assert_not_called()

    def test_runner_rejects_bias_and_repetition_as_session_before_evaluation(self):
        runner = load_har_runner()
        invalid_settings = (
            {"reservoir_params": {"bias_scaling": 0.05}},
            {"parameter_candidates": [{"bias_scaling": 0.05}]},
            {"protocols": ["session_split"]},
        )
        for config in invalid_settings:
            with self.subTest(config=config), patch.object(runner, "run_fusion_comparison") as evaluate:
                with self.assertRaises(ValueError):
                    runner.main(fusion_config=config, loaded_data=({}, [], []))
                evaluate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
