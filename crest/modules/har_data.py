"""Load one room and optionally one distance of high-precision HAR data.

The source tensors have axes (channel, time, range, Doppler).  Their amplitude
is summed over range or Doppler to produce the same eight named time-feature
maps used in the Soli comparison.  No binarization, FFT, logarithm, or dataset
normalization is applied here; train-only standardization belongs to FusionESN.
"""

from collections import Counter
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
import re

import numpy as np


HIGH_PRECISION_DIR = "Human activity recognition V2.0_Clipping"
HAR_TENSOR_SHAPE = (4, 128, 32, 32)
HAR_DISTANCE_METERS = {1: 1.5, 2: 3.5, 3: 5.5}
_FILE_NAME = re.compile(
    r"S(?P<subject>[1-3])_H(?P<room>[1-4])_A(?P<action>10|[1-9])_"
    r"D(?P<distance>[1-3])_(?P<repetition>20|1[0-9]|[1-9])\.npy"
)
_LFS_HEADER = b"version https://git-lfs.github.com/spec/v1\n"


@dataclass(frozen=True)
class HARSample:
    """One recording, with numeric acquisition identifiers from its filename."""

    path: Path
    subject: int
    room: int
    action: int
    distance: int
    repetition: int

    @property
    def sample_id(self):
        return self.subject, self.room, self.action, self.distance, self.repetition

    def metadata(self):
        return dict(filename=self.path.name, subject=self.subject, room=self.room,
                    action=self.action, distance=self.distance,
                    repetition=self.repetition)


def _validate_room(room):
    # H4 occurs in the files, although only H1-H3 are described in the README.
    # Requiring an explicit room avoids silently assigning H4 an environment.
    if not isinstance(room, Integral) or isinstance(room, bool) or room not in (1, 2, 3, 4):
        raise ValueError("room must be an explicit integer in 1, 2, 3, or 4")
    return int(room)


def _validate_distance(distance):
    if (not isinstance(distance, Integral) or isinstance(distance, bool)
            or distance not in HAR_DISTANCE_METERS):
        raise ValueError("distance must be an integer in 1, 2, or 3")
    return int(distance)


def _high_precision_path(base_dir):
    """Resolve only the documented 16-bit folder, never the paired 1-bit data."""
    base_dir = Path(base_dir)
    data_dir = base_dir / HIGH_PRECISION_DIR
    if not data_dir.is_dir():
        raise FileNotFoundError(f"16-bit HAR directory not found: {data_dir}")
    return data_dir


def discover_har_files(base_dir, *, room=None, distance=None):
    """Return recordings in deterministic numeric order without loading arrays.

    ``base_dir`` is the dataset repository root.  Omitting ``room`` is useful
    only for inventory/download tooling; the data loader always requires one.
    ``distance`` can additionally select D1=1.5 m, D2=3.5 m, or D3=5.5 m.
    Invalid filenames or repeated acquisition IDs fail before any experiment.
    """
    if room is not None:
        room = _validate_room(room)
    if distance is not None:
        distance = _validate_distance(distance)
    data_dir = _high_precision_path(base_dir)
    samples, seen = [], {}
    for path in data_dir.rglob("*.npy"):
        match = _FILE_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError(f"Invalid HAR recording filename: {path}")
        sample = HARSample(path, **{key: int(value) for key, value in match.groupdict().items()})
        if sample.sample_id in seen:
            raise ValueError(f"Duplicate HAR acquisition ID: {seen[sample.sample_id]} and {path}")
        seen[sample.sample_id] = path
        if ((room is None or sample.room == room)
                and (distance is None or sample.distance == distance)):
            samples.append(sample)
    samples.sort(key=lambda sample: sample.sample_id)
    if not samples:
        suffix = "" if room is None else f" for room H{room}"
        if distance is not None:
            suffix += f" at distance D{distance} ({HAR_DISTANCE_METERS[distance]} m)"
        raise ValueError(f"No 16-bit HAR recordings found{suffix} in {data_dir}")
    return samples


def read_lfs_pointer(path):
    """Read a Git LFS pointer's SHA256/size, or return None for an actual array."""
    path = Path(path)
    with path.open("rb") as stream:
        header = stream.read(512)
    if not header.startswith(_LFS_HEADER):
        return None
    try:
        lines = header.decode("ascii").splitlines()
    except UnicodeDecodeError as exc:
        raise ValueError(f"Malformed Git LFS pointer: {path}") from exc
    oids = [line.removeprefix("oid sha256:") for line in lines if line.startswith("oid sha256:")]
    sizes = [line.removeprefix("size ") for line in lines if line.startswith("size ")]
    if (len(oids) != 1 or not re.fullmatch(r"[0-9a-f]{64}", oids[0])
            or len(sizes) != 1 or not sizes[0].isdigit() or int(sizes[0]) <= 0):
        raise ValueError(f"Malformed Git LFS pointer: {path}")
    return dict(oid=oids[0], size=int(sizes[0]))


def inspect_har_dataset(base_dir):
    """Inventory 16-bit room membership and missing Git LFS payloads."""
    samples = discover_har_files(base_dir)
    rooms = {}
    for room in sorted({sample.room for sample in samples}):
        selected = [sample for sample in samples if sample.room == room]
        status = Counter()
        for sample in selected:
            if read_lfs_pointer(sample.path) is not None:
                status["lfs_pointers"] += 1
            else:
                with sample.path.open("rb") as stream:
                    status["materialized_npy" if stream.read(6) == b"\x93NUMPY" else "invalid_files"] += 1
        rooms[str(room)] = dict(
            n_samples=len(selected),
            class_counts={str(key): value for key, value in sorted(Counter(s.action for s in selected).items())},
            subjects=sorted({sample.subject for sample in selected}),
            distances=sorted({sample.distance for sample in selected}),
            lfs_pointers=status["lfs_pointers"],
            materialized_npy=status["materialized_npy"],
            invalid_files=status["invalid_files"],
        )
    return dict(data_dir=str(_high_precision_path(base_dir).resolve()),
                precision="16-bit source", n_samples=len(samples),
                room_counts={room: row["n_samples"] for room, row in rooms.items()},
                lfs_pointers=sum(row["lfs_pointers"] for row in rooms.values()),
                rooms=rooms)


class HARDataLoader:
    """Create Soli-compatible maps from one room and an optional distance."""

    def __init__(self, base_dir, *, room, channels=(0, 1, 2, 3), distance=None):
        self.base_dir = Path(base_dir)
        self.room = _validate_room(room)
        self.distance = None if distance is None else _validate_distance(distance)
        self.channels = tuple(channels)
        if (not self.channels or len(set(self.channels)) != len(self.channels)
                or any(not isinstance(channel, Integral) or isinstance(channel, bool)
                       or channel not in (0, 1, 2, 3) for channel in self.channels)):
            raise ValueError("channels must be nonempty, unique integers from 0, 1, 2, 3")
        self.data_dir = _high_precision_path(self.base_dir)
        self.data_manifest = None

    def load_all_data(self):
        """Return ``(maps, action_labels, metadata)`` without filtering recordings.

        Labels preserve the documented action IDs 1-10.  Missing payloads and
        malformed arrays are errors, rather than silently reducing the dataset.
        """
        samples = discover_har_files(self.base_dir, room=self.room, distance=self.distance)
        missing = [sample.path for sample in samples if read_lfs_pointer(sample.path) is not None]
        if missing:
            raise FileNotFoundError(
                f"Room H{self.room}"
                + ("" if self.distance is None else f" at distance D{self.distance} ({HAR_DISTANCE_METERS[self.distance]} m)")
                + f" has {len(missing)} Git LFS pointers without array payloads. "
                f"Download the 16-bit recordings before evaluation. First missing file: {missing[0]}"
            )
        maps = {name: [] for channel in self.channels
                for name in (f"DTM_ch{channel}", f"RTM_ch{channel}")}
        dtypes = Counter()
        for sample in samples:
            try:
                tensor = np.load(sample.path, allow_pickle=False)
            except (ValueError, OSError, EOFError) as exc:
                raise ValueError(f"Cannot load HAR array: {sample.path}") from exc
            if not isinstance(tensor, np.ndarray):
                if hasattr(tensor, "close"):
                    tensor.close()
                raise ValueError(f"HAR recording must contain a single NumPy array: {sample.path}")
            if tensor.shape != HAR_TENSOR_SHAPE:
                raise ValueError(f"HAR tensor {sample.path} has shape {tensor.shape}; expected {HAR_TENSOR_SHAPE}")
            if not np.issubdtype(tensor.dtype, np.number) or np.issubdtype(tensor.dtype, np.bool_):
                raise ValueError(f"HAR tensor {sample.path} must have a numeric, non-boolean dtype")
            if not np.isfinite(tensor).all():
                raise ValueError(f"HAR tensor contains non-finite values: {sample.path}")
            dtypes[str(tensor.dtype)] += 1
            # Cast signed integers before abs, so -32768 cannot overflow int16.
            amplitude = np.abs(tensor.astype(np.complex128 if np.iscomplexobj(tensor) else np.float64))
            dtm = amplitude.sum(axis=2, dtype=np.float64)
            rtm = amplitude.sum(axis=3, dtype=np.float64)
            for channel in self.channels:
                dtm_map = dtm[channel].astype(np.float32)
                rtm_map = rtm[channel].astype(np.float32)
                if not np.isfinite(dtm_map).all() or not np.isfinite(rtm_map).all():
                    raise ValueError(f"HAR projected features contain non-finite values: {sample.path}")
                maps[f"DTM_ch{channel}"].append(dtm_map)
                maps[f"RTM_ch{channel}"].append(rtm_map)
        metadata = [sample.metadata() for sample in samples]
        labels = np.asarray([sample.action for sample in samples], dtype=int)
        self.data_manifest = dict(
            data_dir=str(self.data_dir.resolve()), precision="16-bit source", room=self.room,
            distance=self.distance,
            distance_meters=None if self.distance is None else HAR_DISTANCE_METERS[self.distance],
            n_samples=len(samples), tensor_shape=list(HAR_TENSOR_SHAPE),
            source_dtype_counts=dict(sorted(dtypes.items())),
            feature_dtype="float32", map_shape=[128, 32], map_order=list(maps),
            feature_extraction="abs(tensor); DTM=sum over range; RTM=sum over Doppler",
            channels=list(self.channels),
            subjects=sorted({sample.subject for sample in samples}),
            distances=sorted({sample.distance for sample in samples}),
            class_counts={str(key): value for key, value in sorted(Counter(labels.tolist()).items())},
        )
        return maps, labels, metadata
