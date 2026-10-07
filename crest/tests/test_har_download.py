"""Verify offline that HAR downloads replace only verified, selected blobs."""

import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from modules.har_data import HIGH_PRECISION_DIR
from modules.har_download import _fetch_blob, download_har_room


class HARDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base_dir = Path(self.temporary.name)
        self.data_dir = self.base_dir / HIGH_PRECISION_DIR
        self.data_dir.mkdir()
        self.payload = b"\x93NUMPYverified high precision test payload"
        self.pointer = dict(oid=hashlib.sha256(self.payload).hexdigest(), size=len(self.payload))

    def save_pointer(self, filename, directory=None):
        directory = self.data_dir if directory is None else directory
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / filename
        path.write_text("version https://git-lfs.github.com/spec/v1\n"
                        f"oid sha256:{self.pointer['oid']}\nsize {self.pointer['size']}\n",
                        encoding="ascii")
        return path

    def test_verified_blob_replaces_pointer_and_uses_committed_high_precision_url(self):
        path = self.save_pointer("S1_H1_A1_D1_1.npy")
        with patch("modules.har_download.urlopen", return_value=io.BytesIO(self.payload)) as fetch:
            downloaded = _fetch_blob(path, self.pointer, "a" * 40)
        self.assertEqual(downloaded, len(self.payload))
        self.assertEqual(path.read_bytes(), self.payload)
        self.assertFalse(path.with_name(path.name + ".download-part").exists())
        url = fetch.call_args.args[0]
        self.assertIn("/" + "a" * 40 + "/", url)
        self.assertIn("Human%20activity%20recognition%20V2.0_Clipping/", url)
        self.assertNotIn("1bit", url)

    def test_hash_or_size_mismatch_keeps_pointer_and_removes_partial_file(self):
        for bad_payload in (b"wrong size", b"x" * len(self.payload)):
            with self.subTest(payload=bad_payload):
                path = self.save_pointer("S1_H1_A1_D1_1.npy")
                original = path.read_bytes()
                with patch("modules.har_download.urlopen", return_value=io.BytesIO(bad_payload)):
                    with self.assertRaisesRegex(RuntimeError, "integrity check failed"):
                        _fetch_blob(path, self.pointer, "a" * 40, retries=1)
                self.assertEqual(path.read_bytes(), original)
                self.assertFalse(path.with_name(path.name + ".download-part").exists())

    def test_retry_after_network_failure_does_not_overwrite_pointer_early(self):
        path = self.save_pointer("S1_H1_A1_D1_1.npy")
        original = path.read_bytes()

        def fail_then_succeed(*args, **kwargs):
            self.assertEqual(path.read_bytes(), original)
            if fail_then_succeed.calls == 0:
                fail_then_succeed.calls += 1
                raise OSError("temporary network failure")
            return io.BytesIO(self.payload)

        fail_then_succeed.calls = 0
        with patch("modules.har_download.urlopen", side_effect=fail_then_succeed), \
                patch("modules.har_download.time.sleep"):
            self.assertEqual(_fetch_blob(path, self.pointer, "a" * 40, retries=2), len(self.payload))
        self.assertEqual(path.read_bytes(), self.payload)

    def test_room_download_excludes_other_rooms_and_one_bit_and_preserves_manifest(self):
        chosen = [self.save_pointer(f"S1_H1_A1_D1_{repeat}.npy") for repeat in (1, 2)]
        other_room = self.save_pointer("S1_H2_A1_D1_1.npy")
        one_bit = self.save_pointer("S1_H1_A1_D1_1.npy",
                                    self.base_dir / "Human Activity_1bit v2.0_Clipping")
        untouched = {path: path.read_bytes() for path in (other_room, one_bit)}

        def download(path, pointer, commit):
            self.assertEqual(pointer, self.pointer)
            path.write_bytes(self.payload)
            return len(self.payload)

        with patch("modules.har_download.subprocess.check_output", return_value="a" * 40 + "\n"), \
                patch("modules.har_download._fetch_blob", side_effect=download) as fetch, \
                patch("builtins.print"):
            manifest = download_har_room(self.base_dir, room=1, workers=1)
            self.assertEqual(fetch.call_count, 2)
            self.assertEqual({call.args[0] for call in fetch.call_args_list}, set(chosen))
            # A completed run should reuse the recorded original hashes.
            resumed = download_har_room(self.base_dir, room=1, workers=1)
            self.assertEqual(fetch.call_count, 2)
        self.assertEqual(manifest, resumed)
        self.assertEqual(manifest["room"], 1)
        self.assertEqual(set(manifest["blobs"]), {path.name for path in chosen})
        self.assertEqual(manifest["blobs"][chosen[0].name], self.pointer)
        persisted = json.loads((self.base_dir / "results" / "download_H1_manifest.json").read_text())
        self.assertEqual(manifest, persisted)
        for path, original in untouched.items():
            self.assertEqual(path.read_bytes(), original)

    def test_distance_download_excludes_other_distances_and_keeps_separate_provenance(self):
        chosen = self.save_pointer("S1_H2_A1_D1_1.npy")
        other_distance = self.save_pointer("S1_H2_A1_D2_1.npy")
        other_room = self.save_pointer("S1_H3_A1_D1_1.npy")
        untouched = {path: path.read_bytes() for path in (other_distance, other_room)}

        def download(path, pointer, commit):
            self.assertEqual(path, chosen)
            path.write_bytes(self.payload)
            return len(self.payload)

        with patch("modules.har_download.subprocess.check_output", return_value="a" * 40 + "\n"), \
                patch("modules.har_download._fetch_blob", side_effect=download) as fetch, \
                patch("builtins.print"):
            manifest = download_har_room(self.base_dir, room=2, distance=1, workers=1)
            resumed = download_har_room(self.base_dir, room=2, distance=1, workers=1)
        self.assertEqual(fetch.call_count, 1)
        self.assertEqual(manifest, resumed)
        self.assertEqual((manifest["room"], manifest["distance"], manifest["n_samples"]), (2, 1, 1))
        self.assertEqual(set(manifest["blobs"]), {chosen.name})
        self.assertTrue((self.base_dir / "results/download_H2_D1_manifest.json").exists())
        self.assertFalse((self.base_dir / "results/download_H2_manifest.json").exists())
        for path, original in untouched.items():
            self.assertEqual(path.read_bytes(), original)

    def test_materialized_distance_inherits_room_hashes_and_repairs_empty_manifest(self):
        files = [self.save_pointer(f"S1_H1_A1_D{distance}_1.npy") for distance in (1, 2)]

        def download(path, pointer, commit):
            path.write_bytes(self.payload)
            return len(self.payload)

        with patch("modules.har_download.subprocess.check_output", return_value="a" * 40 + "\n"), \
                patch("modules.har_download._fetch_blob", side_effect=download) as fetch, \
                patch("builtins.print"):
            room_manifest = download_har_room(self.base_dir, room=1, workers=1)
            distance_manifest = download_har_room(self.base_dir, room=1, distance=2, workers=1)
            scoped_path = self.base_dir / "results/download_H1_D2_manifest.json"
            # Older distance-specific calls could record no hashes when the
            # payloads were already fetched by a room-wide download.
            scoped_path.write_text(json.dumps(dict(distance_manifest, blobs={})))
            repaired = download_har_room(self.base_dir, room=1, distance=2, workers=1)
        self.assertEqual(fetch.call_count, 2)
        self.assertEqual(repaired, distance_manifest)
        self.assertEqual(distance_manifest["distance"], 2)
        self.assertEqual(distance_manifest["n_samples"], 1)
        self.assertEqual(distance_manifest["blobs"], {files[1].name: self.pointer})
        self.assertEqual(json.loads(scoped_path.read_text()), distance_manifest)
        self.assertEqual(json.loads((self.base_dir / "results/download_H1_manifest.json").read_text()),
                         room_manifest)

    def test_distance_does_not_inherit_room_hashes_from_a_different_commit(self):
        path = self.save_pointer("S1_H1_A1_D2_1.npy")
        path.write_bytes(self.payload)
        manifest_dir = self.base_dir / "results"
        manifest_dir.mkdir()
        (manifest_dir / "download_H1_manifest.json").write_text(json.dumps(dict(
            dataset="HAR-mmWave 16-bit", room=1, source_commit="b" * 40,
            n_samples=1, blobs={path.name: self.pointer})))
        with patch("modules.har_download.subprocess.check_output", return_value="a" * 40 + "\n"), \
                patch("modules.har_download._fetch_blob") as fetch, patch("builtins.print"):
            manifest = download_har_room(self.base_dir, room=1, distance=2, workers=1)
        fetch.assert_not_called()
        self.assertEqual(manifest["source_commit"], "a" * 40)
        self.assertEqual(manifest["blobs"], {})


if __name__ == "__main__":
    unittest.main()
