"""Controlled ESN comparisons for reservoir topology and fusion stage.

Each example is a mapping ``{map_name: [time_by_feature_array, ...]}``.
All models use final reservoir states and an intercept-free ridge readout:
``||Y - X W||**2 + regularization * ||W||**2``, using the existing RR_L
readout implementation. The parallel models split ``total_nodes`` equally between maps.
Existing ESN implementations are deliberately left unchanged.
"""

from collections.abc import Mapping
from numbers import Integral, Real

import numpy as np
from scipy.linalg import eigvals
from scipy.sparse import csr_matrix
from scipy.special import softmax

from .readouts import fit_ridge_readout


ARCHITECTURES = (
    "single_map",
    "single_early",
    "parallel_early",
    "parallel_intermediate",
    "parallel_late",
)
LATE_FUSION_METHODS = ("mean", "geometric", "product", "max")


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _finite_real(value, name, *, minimum=0.0, strictly_positive=False):
    if isinstance(value, bool) or not isinstance(value, Real) or not np.isfinite(value):
        raise ValueError(f"{name} must be a finite real number")
    if value < minimum or (strictly_positive and value <= minimum):
        comparison = ">" if strictly_positive else ">="
        raise ValueError(f"{name} must be {comparison} {minimum}")
    return float(value)


def _fusion_weights(weights, n_branches, method):
    if weights is None:
        return np.full(n_branches, 1.0 / n_branches)
    if method not in ("mean", "geometric"):
        raise ValueError("fusion_weights are supported only for mean and geometric fusion")
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (n_branches,):
        raise ValueError(f"fusion_weights must contain {n_branches} values")
    if not np.all(np.isfinite(weights)) or np.any(weights <= 0):
        raise ValueError("fusion_weights must be finite and strictly positive")
    # Scaling first avoids overflow when users supply large finite weights.
    weights = weights / weights.max()
    return weights / weights.sum()


def fuse_probabilities(probabilities, method="mean", weights=None):
    """Combine branch probabilities of shape (branches, examples, classes).

    ``mean`` is an arithmetic linear pool, ``geometric`` a normalized geometric
    pool, ``product`` a normalized product, and ``max`` the normalized classwise
    maximum. Product and geometric pooling are evaluated in log space. Fixed
    positive weights may be supplied for mean or geometric pooling; this helper
    never estimates weights from evaluation labels.
    """
    if method not in LATE_FUSION_METHODS:
        raise ValueError(f"Unknown late fusion method: {method!r}")
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 3 or probabilities.shape[0] < 1 or probabilities.shape[2] < 1:
        raise ValueError("probabilities must have shape (branches, examples, classes)")
    if not np.all(np.isfinite(probabilities)) or np.any(probabilities < 0):
        raise ValueError("probabilities must be finite and nonnegative")
    totals = probabilities.sum(axis=2, keepdims=True)
    if np.any(totals <= 0) or not np.all(np.isfinite(totals)):
        raise ValueError("each branch probability vector must have a positive finite sum")
    probabilities = probabilities / totals
    branch_weights = _fusion_weights(weights, probabilities.shape[0], method)
    if method == "mean":
        pooled = np.tensordot(branch_weights, probabilities, axes=(0, 0))
    elif method == "max":
        pooled = probabilities.max(axis=0)
    else:
        log_probabilities = np.log(np.clip(probabilities, np.finfo(float).tiny, 1.0))
        if method == "geometric":
            log_pool = np.tensordot(branch_weights, log_probabilities, axes=(0, 0))
        else:
            log_pool = log_probabilities.sum(axis=0)
        return softmax(log_pool, axis=1)
    return pooled / pooled.sum(axis=1, keepdims=True)


class _FairReservoir:
    """A local-RNG ESN whose recurrent draws do not depend on input width."""

    def __init__(self, n_nodes, n_inputs, model, branch_index):
        self.n_nodes = n_nodes
        self.n_inputs = n_inputs
        self.leakage_rate = model.leakage_rate
        streams = np.random.SeedSequence([model.random_state, branch_index]).spawn(3)
        recurrent_rng, input_rng, bias_rng = [np.random.default_rng(s) for s in streams]
        self.W_res = recurrent_rng.normal(size=(n_nodes, n_nodes))
        self.W_res *= recurrent_rng.random((n_nodes, n_nodes)) < model.density
        if model.spectral_radius == 0:
            self.W_res.fill(0)
        else:
            radius = np.max(np.abs(eigvals(self.W_res, check_finite=False)))
            if radius == 0:
                # Very small, sparse reservoirs can have no cycle. Add one
                # deterministic self connection so the requested radius exists.
                self.W_res[0, 0] = 1.0
                radius = 1.0
            self.W_res *= model.spectral_radius / radius
        self.W_in = input_rng.normal(
            scale=model.input_scaling / np.sqrt(n_inputs), size=(n_nodes, n_inputs)
        )
        self.W_bias = bias_rng.uniform(-model.bias_scaling, model.bias_scaling, n_nodes)
        self._recurrent = csr_matrix(self.W_res)

    def transform_sequences(self, sequences):
        states = np.empty((len(sequences), self.n_nodes), dtype=float)
        for i, sequence in enumerate(sequences):
            state = np.zeros(self.n_nodes, dtype=float)
            # The input projection is batched; recurrent updates remain causal.
            projected = sequence @ self.W_in.T
            for drive in projected:
                candidate = np.tanh(drive + self._recurrent @ state + self.W_bias)
                state = (1.0 - self.leakage_rate) * state + self.leakage_rate * candidate
            states[i] = state
        return states


class FusionESN:
    """Five ESN architectures with a fixed node and linear readout budget.

    Parameters use a shared interpretation across architectures. Input scaling
    is divided by sqrt(input width), and input standardization is fitted using
    training time frames only. ``single_map`` uses ``map_name`` (or the sole map
    when omitted). Parallel architectures use one branch per input map and
    require ``total_nodes`` to be divisible by the number of maps.

    ``extract_features`` always returns the concatenated final states with
    exactly ``total_nodes`` columns. ``reservoirs_`` and ``readouts_`` are lists
    in map/branch order; readout matrices have shape (nodes, classes). For late
    fusion the total number of fitted readout coefficients is still N * C.
    """

    def __init__(self, architecture, total_nodes=400, regularization=0.001,
                 spectral_radius=0.95, input_scaling=0.2, density=0.1,
                 leakage_rate=0.05, bias_scaling=0.0, random_state=42,
                 map_name=None, late_fusion="mean", fusion_weights=None,
                 temperature=1.0, standardize_inputs=True):
        if architecture not in ARCHITECTURES:
            raise ValueError(f"Unknown architecture: {architecture!r}")
        if late_fusion not in LATE_FUSION_METHODS:
            raise ValueError(f"Unknown late fusion method: {late_fusion!r}")
        if map_name is not None and (not isinstance(map_name, str) or not map_name):
            raise ValueError("map_name must be a nonempty string or None")
        if map_name is not None and architecture != "single_map":
            raise ValueError("map_name is only supported for single_map")
        if fusion_weights is not None and architecture != "parallel_late":
            raise ValueError("fusion_weights are only supported for parallel_late")
        if not isinstance(standardize_inputs, (bool, np.bool_)):
            raise ValueError("standardize_inputs must be boolean")
        if (isinstance(random_state, bool) or not isinstance(random_state, Integral)
                or random_state < 0):
            raise ValueError("random_state must be a nonnegative integer")
        self.architecture = architecture
        self.total_nodes = _positive_integer(total_nodes, "total_nodes")
        self.regularization = _finite_real(regularization, "regularization")
        self.spectral_radius = _finite_real(spectral_radius, "spectral_radius")
        self.input_scaling = _finite_real(input_scaling, "input_scaling")
        self.density = _finite_real(density, "density", strictly_positive=True)
        self.leakage_rate = _finite_real(leakage_rate, "leakage_rate", strictly_positive=True)
        if self.density > 1 or self.leakage_rate > 1:
            raise ValueError("density and leakage_rate must be <= 1")
        self.bias_scaling = _finite_real(bias_scaling, "bias_scaling")
        self.random_state = int(random_state)
        self.map_name = map_name
        self.late_fusion = late_fusion
        self.fusion_weights = fusion_weights
        self.temperature = _finite_real(temperature, "temperature", strictly_positive=True)
        self.standardize_inputs = bool(standardize_inputs)

    def _validate_maps(self, maps, *, fitting):
        if not isinstance(maps, Mapping) or not maps:
            raise ValueError("maps must be a nonempty mapping of names to sequence lists")
        if any(not isinstance(name, str) or not name for name in maps):
            raise ValueError("all map names must be nonempty strings")
        if fitting:
            if self.architecture == "single_map":
                name = self.map_name
                if name is None:
                    if len(maps) != 1:
                        raise ValueError("single_map requires map_name when several maps are provided")
                    name = next(iter(maps))
                if name not in maps:
                    raise ValueError(f"Missing selected map: {name!r}")
                names = (name,)
            else:
                names = tuple(maps)
        else:
            names = self.map_names_
            if self.architecture == "single_map":
                if names[0] not in maps:
                    raise ValueError(f"Missing selected map: {names[0]!r}")
            elif set(maps) != set(names):
                raise ValueError("prediction maps must match the training map names")
        validated, widths, sample_count, lengths = {}, {}, None, None
        for name in names:
            try:
                sequences = list(maps[name])
            except TypeError as exc:
                raise ValueError(f"Map {name!r} must contain a sequence list") from exc
            if fitting and not sequences:
                raise ValueError("training maps must contain at least one example")
            if sample_count is None:
                sample_count = len(sequences)
            elif len(sequences) != sample_count:
                raise ValueError("all maps must contain the same number of examples")
            expected_width = None if fitting else self.n_features_in_[name]
            converted = []
            map_lengths = []
            for index, sequence in enumerate(sequences):
                try:
                    sequence = np.asarray(sequence, dtype=float)
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"Map {name!r}, example {index} must be numeric") from exc
                if sequence.ndim != 2 or 0 in sequence.shape:
                    raise ValueError(f"Map {name!r}, example {index} must be a nonempty 2D array")
                if not np.all(np.isfinite(sequence)):
                    raise ValueError(f"Map {name!r}, example {index} contains nonfinite values")
                if expected_width is None:
                    expected_width = sequence.shape[1]
                elif sequence.shape[1] != expected_width:
                    raise ValueError(f"Feature width changed for map {name!r}")
                converted.append(sequence)
                map_lengths.append(sequence.shape[0])
            if lengths is None:
                lengths = map_lengths
            elif map_lengths != lengths:
                raise ValueError("time lengths must be aligned across maps for every example")
            validated[name] = converted
            widths[name] = expected_width
        return validated, names, widths, sample_count

    @staticmethod
    def _fit_scaler(sequences):
        # Merge per-sequence moments without concatenating an entire dataset.
        count = 0
        mean = np.zeros(sequences[0].shape[1], dtype=float)
        second_moment = np.zeros_like(mean)
        for sequence in sequences:
            batch_count = len(sequence)
            batch_mean = sequence.mean(axis=0)
            batch_second_moment = ((sequence - batch_mean) ** 2).sum(axis=0)
            total_count = count + batch_count
            delta = batch_mean - mean
            mean += delta * (batch_count / total_count)
            second_moment += batch_second_moment + delta ** 2 * (count * batch_count / total_count)
            count = total_count
        scale = np.sqrt(np.maximum(second_moment / count, 0.0))
        scale[scale == 0] = 1.0
        if not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
            raise ValueError("training input magnitudes exceed stable standardization range")
        return {"mean": mean, "scale": scale}

    def _normalized_maps(self, validated):
        if not self.standardize_inputs:
            return validated
        normalized = {}
        for name, sequences in validated.items():
            scaler = self.scalers_[name]
            normalized[name] = [(sequence - scaler["mean"]) / scaler["scale"]
                                for sequence in sequences]
            if any(not np.all(np.isfinite(seq)) for seq in normalized[name]):
                raise ValueError(f"Standardized map {name!r} contains nonfinite values")
        return normalized

    def _branch_features(self, validated):
        maps = self._normalized_maps(validated)
        if self.architecture in ("single_early", "parallel_early"):
            count = len(maps[self.map_names_[0]])
            combined = [np.concatenate([maps[name][i] for name in self.map_names_], axis=1)
                        for i in range(count)]
            return [reservoir.transform_sequences(combined) for reservoir in self.reservoirs_]
        return [reservoir.transform_sequences(maps[name])
                for name, reservoir in zip(self.map_names_, self.reservoirs_)]

    def _ridge(self, features, targets):
        # RR_L stores W_out as (classes, nodes); feature-based APIs use its transpose.
        return fit_ridge_readout(features, targets, self.regularization).T

    def fit(self, maps, y):
        """Fit training-only preprocessing and ridge readout(s), returning self."""
        validated, names, widths, n_samples = self._validate_maps(maps, fitting=True)
        y = np.asarray(y)
        if y.ndim != 1 or len(y) != n_samples:
            raise ValueError("y must be a 1D label array matching the number of examples")
        if y.dtype.kind in "fc" and not np.all(np.isfinite(y)):
            raise ValueError("labels must be finite")
        if y.dtype.kind == "O" and any(label is None for label in y):
            raise ValueError("labels must not contain None")
        try:
            classes, encoded = np.unique(y, return_inverse=True)
        except TypeError as exc:
            raise ValueError("labels must have mutually comparable scalar values") from exc
        n_branches = len(names) if self.architecture.startswith("parallel_") else 1
        if self.total_nodes < n_branches or self.total_nodes % n_branches:
            raise ValueError("total_nodes must be >= map count and divisible by map count")
        if self.architecture == "parallel_late":
            branch_weights = _fusion_weights(self.fusion_weights, n_branches, self.late_fusion)
        else:
            branch_weights = None
        self.map_names_ = names
        self.n_features_in_ = widths
        self.classes_ = classes
        self.node_counts_ = tuple([self.total_nodes // n_branches] * n_branches)
        self.fusion_weights_ = branch_weights
        self.scalers_ = {}
        for name in names:
            self.scalers_[name] = self._fit_scaler(validated[name]) if self.standardize_inputs else {
                "mean": np.zeros(widths[name]), "scale": np.ones(widths[name])
            }
        if self.architecture in ("single_early", "parallel_early"):
            input_widths = [sum(widths.values())] * n_branches
        else:
            input_widths = [widths[name] for name in names]
        self.reservoirs_ = [_FairReservoir(nodes, width, self, index)
                            for index, (nodes, width) in enumerate(zip(self.node_counts_, input_widths))]
        branches = self._branch_features(validated)
        targets = np.eye(len(classes))[encoded]
        readout_features = branches if self.architecture == "parallel_late" else [np.hstack(branches)]
        self.readouts_ = [self._ridge(features, targets) for features in readout_features]
        return self

    def _check_fitted(self):
        if not hasattr(self, "readouts_"):
            raise ValueError("FusionESN must be fitted before extracting features or predicting")

    def extract_features(self, maps):
        """Return concatenated final reservoir states, shape (examples, N)."""
        self._check_fitted()
        validated, _, _, _ = self._validate_maps(maps, fitting=False)
        return np.hstack(self._branch_features(validated))

    def predict_proba(self, maps):
        """Return softmax-derived probabilities in ``classes_`` order."""
        return self.predict_proba_from_features(self.extract_features(maps))

    def _validate_features(self, features):
        self._check_fitted()
        features = np.asarray(features, dtype=float)
        if features.ndim != 2 or features.shape[1] != self.total_nodes:
            raise ValueError(f"features must have shape (examples, {self.total_nodes})")
        if not np.all(np.isfinite(features)):
            raise ValueError("features must be finite")
        return features

    def branch_probabilities_from_features(self, features):
        """Return late branch probabilities (branches, examples, classes).

        These can be reused to compare fixed fusion rules without repeating
        reservoir extraction or readout fitting.
        """
        features = self._validate_features(features)
        if self.architecture != "parallel_late":
            raise ValueError("branch probabilities require parallel_late")
        branches = np.split(features, np.cumsum(self.node_counts_)[:-1], axis=1)
        return np.stack([softmax(branch @ readout / self.temperature, axis=1)
                         for branch, readout in zip(branches, self.readouts_)])

    def predict_proba_from_features(self, features, *, late_fusion=None):
        """Predict from final states, optionally overriding a late fusion rule."""
        features = self._validate_features(features)
        if self.architecture != "parallel_late":
            if late_fusion is not None:
                raise ValueError("late_fusion override requires parallel_late")
            return softmax(features @ self.readouts_[0] / self.temperature, axis=1)
        method = self.late_fusion if late_fusion is None else late_fusion
        if method not in LATE_FUSION_METHODS:
            raise ValueError(f"Unknown late fusion method: {method!r}")
        if method in ("product", "max"):
            if self.fusion_weights is not None:
                raise ValueError("fusion_weights are supported only for mean and geometric fusion")
            weights = None
        else:
            weights = self.fusion_weights_
        return fuse_probabilities(self.branch_probabilities_from_features(features), method, weights)

    def predict(self, maps):
        """Return original class labels rather than class-column indices."""
        probabilities = self.predict_proba(maps)
        return self.classes_[np.argmax(probabilities, axis=1)]

    @property
    def parameter_counts(self):
        """Report comparable trained budgets and topology-dependent fixed weights."""
        self._check_fitted()
        input_weights = sum(reservoir.W_in.size for reservoir in self.reservoirs_)
        recurrent_weights = sum(reservoir.W_res.size for reservoir in self.reservoirs_)
        readout_weights = sum(readout.size for readout in self.readouts_)
        reservoir_biases = self.total_nodes if self.bias_scaling != 0 else 0
        return {
            "total_nodes": sum(self.node_counts_),
            "readout_features": self.total_nodes,
            "readout_weights": readout_weights,
            "trainable_parameters": readout_weights,
            "input_weights": input_weights,
            "recurrent_weights": recurrent_weights,
            "recurrent_nonzero": int(sum(np.count_nonzero(r.W_res) for r in self.reservoirs_)),
            "reservoir_biases": reservoir_biases,
            "fixed_parameters": input_weights + recurrent_weights + reservoir_biases,
        }
