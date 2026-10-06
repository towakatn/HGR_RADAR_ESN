"""Download only a selected room's high-precision HAR arrays from Git LFS.

Run from crest: python -m modules.har_download --room 1
Only the 16-bit folder is accessed. Each downloaded blob is checked against
its committed LFS SHA256 and byte count before replacing its pointer file.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import quote
from urllib.request import urlopen

from .har_data import HIGH_PRECISION_DIR, discover_har_files, read_lfs_pointer


def _fetch_blob(path, pointer, commit, retries=3):
    relative = f"{HIGH_PRECISION_DIR}/{path.name}"
    url = ("https://media.githubusercontent.com/media/embedded-qjd/"
           f"HAR-Dataset-Project/{commit}/{quote(relative, safe='/')}")
    temporary = path.with_name(path.name + ".download-part")
    last_error = None
    for attempt in range(retries):
        try:
            digest = hashlib.sha256()
            size = 0
            with urlopen(url, timeout=120) as response, temporary.open("wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                    digest.update(chunk)
                    size += len(chunk)
            if size != pointer["size"] or digest.hexdigest() != pointer["oid"]:
                raise ValueError(f"LFS integrity check failed for {path.name}")
            temporary.replace(path)
            return size
        except Exception as exc:
            last_error = exc
            temporary.unlink(missing_ok=True)
            if attempt + 1 < retries:
                time.sleep(attempt + 1)
    raise RuntimeError(f"Could not fetch {path.name}: {type(last_error).__name__}: {last_error}") from last_error


def download_har_room(base_dir, *, room, workers=8):
    """Materialize LFS pointers for one room, preserving other rooms and 1-bit data."""
    base_dir = Path(base_dir).resolve()
    samples = discover_har_files(base_dir, room=room)
    if workers < 1:
        raise ValueError("workers must be positive")
    commit = subprocess.check_output(
        ["git", "-C", str(base_dir), "rev-parse", "HEAD"], text=True
    ).strip()
    pending = []
    for sample in samples:
        path = sample.path
        pointer = read_lfs_pointer(path)
        if pointer is not None:
            pending.append((path, pointer))
    total_bytes = sum(pointer["size"] for _, pointer in pending)
    print(f"HAR H{room}: {len(samples)} 16-bit files, {len(pending)} pending, "
          f"{total_bytes / 1e9:.2f} GB, source {commit}", flush=True)
    manifest_dir = base_dir / "results"
    manifest_dir.mkdir(exist_ok=True)
    manifest_path = manifest_dir / f"download_H{room}_manifest.json"
    previous = {}
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("source_commit") != commit:
            raise ValueError("Existing download manifest refers to a different dataset commit")
    blobs = dict(previous.get("blobs", {}))
    blobs.update({path.name: pointer for path, pointer in pending})
    manifest = dict(dataset="HAR-mmWave 16-bit", room=int(room), source_commit=commit,
                    n_samples=len(samples), blobs=blobs)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    if not pending:
        return manifest
    # Probe one actual blob first so network/permission failures are reported
    # without launching thousands of requests.
    path, pointer = pending[0]
    downloaded = _fetch_blob(path, pointer, commit)
    completed = 1
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_fetch_blob, path, pointer, commit): path
                   for path, pointer in pending[1:]}
        try:
            for future in as_completed(futures):
                downloaded += future.result()
                completed += 1
                if completed % 50 == 0 or completed == len(pending):
                    elapsed = time.perf_counter() - started
                    print(f"Downloaded {completed}/{len(pending)} | {downloaded / 1e9:.2f} GB | "
                          f"{elapsed:.0f}s", flush=True)
        except Exception:
            for future in futures:
                future.cancel()
            raise
    return manifest


def cli(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", type=int, choices=(1, 2, 3, 4), required=True)
    parser.add_argument("--base-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "HAR-Dataset-Project")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args(argv)
    return download_har_room(args.base_dir, room=args.room, workers=args.workers)


if __name__ == "__main__":
    cli()
