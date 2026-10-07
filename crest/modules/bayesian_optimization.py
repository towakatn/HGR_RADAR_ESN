"""Fixed-budget Gaussian-process optimization for controlled fusion comparisons.

Only reservoir dynamics, temperature, and input standardization are searched.
Node counts, ridge regularization, reservoir seeds, and biases stay outside the
search space. The supplied baseline is evaluated as the first budgeted trial.
Uses the existing SciPy / scikit-learn dependencies; no optimizer service needed.
"""

from collections.abc import Mapping
from copy import deepcopy
from numbers import Integral, Real
import warnings

import numpy as np
from scipy.special import ndtr
from scipy.stats import qmc
from sklearn.exceptions import ConvergenceWarning
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel


DEFAULT_N_TRIALS = 60
DEFAULT_N_INITIAL_POINTS = 12
DEFAULT_SEARCH_SPACE = {
    "spectral_radius": {"low": 0.1, "high": 1.5, "scale": "log"},
    "input_scaling": {"low": 0.01, "high": 2.0, "scale": "log"},
    "density": {"low": 0.01, "high": 1.0, "scale": "log"},
    "leakage_rate": {"low": 0.001, "high": 1.0, "scale": "log"},
    "temperature": {"low": 0.1, "high": 10.0, "scale": "log"},
    "standardize_inputs": {"choices": [False, True]},
}


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def validate_search_space(search_space=None):
    """Merge domain overrides, rejecting fixed parameters and invalid bounds."""
    space = deepcopy(DEFAULT_SEARCH_SPACE)
    if search_space is not None:
        if not isinstance(search_space, Mapping) or set(search_space) - set(space):
            raise ValueError(f"search_space keys must be among {sorted(space)}; "
                             "nodes, regularization, seeds and bias are fixed")
        space.update(deepcopy(search_space))
    for key, domain in space.items():
        if not isinstance(domain, Mapping):
            raise ValueError(f"{key} search domain must be a dictionary")
        if key == "standardize_inputs":
            choices = domain.get("choices", ())
            if (set(domain) != {"choices"} or not isinstance(choices, (list, tuple))
                    or len(choices) != 2 or any(type(value) is not bool for value in choices)
                    or choices[0] == choices[1]):
                raise ValueError("standardize_inputs choices must be [False, True] or [True, False]")
            space[key] = {"choices": list(choices)}
            continue
        if set(domain) != {"low", "high", "scale"}:
            raise ValueError(f"{key} domain requires low, high and scale")
        low, high = domain["low"], domain["high"]
        if any(isinstance(value, bool) or not isinstance(value, Real)
               or not np.isfinite(value) for value in (low, high)):
            raise ValueError(f"{key} bounds must be finite real numbers")
        positive = key in ("temperature", "density", "leakage_rate")
        if low < 0 or low >= high or (positive and low == 0):
            raise ValueError(f"{key} bounds must satisfy {'0 <' if positive else '0 <='} low < high")
        if key in ("density", "leakage_rate") and high > 1:
            raise ValueError(f"{key} high must be <= 1")
        if domain["scale"] not in ("linear", "log"):
            raise ValueError(f"{key} scale must be linear or log")
        if domain["scale"] == "log" and low <= 0:
            raise ValueError(f"{key} logarithmic low must be > 0")
        space[key] = dict(low=float(low), high=float(high), scale=domain["scale"])
    return space


def _encode(parameters, space):
    point = []
    for name, domain in space.items():
        if name not in parameters:
            raise ValueError(f"base_parameters must supply {name}")
        value = parameters[name]
        if "choices" in domain:
            if not isinstance(value, (bool, np.bool_)):
                raise ValueError(f"{name} must be boolean")
            point.append(float(domain["choices"].index(bool(value))))
            continue
        if (isinstance(value, bool) or not isinstance(value, Real)
                or not np.isfinite(value) or not domain["low"] <= value <= domain["high"]):
            raise ValueError(f"baseline {name} must be inside its search bounds")
        low, high = domain["low"], domain["high"]
        if domain["scale"] == "log":
            value, low, high = np.log(value), np.log(low), np.log(high)
        point.append(float((value - low) / (high - low)))
    return np.asarray(point)


def validate_baseline(base_parameters, search_space=None):
    """Validate bounds against the warm start before any model is fitted."""
    space = validate_search_space(search_space)
    _encode(base_parameters, space)
    return space


def _decode(point, space):
    parameters = {}
    for value, (name, domain) in zip(point, space.items()):
        if "choices" in domain:
            parameters[name] = domain["choices"][int(value >= 0.5)]
        elif domain["scale"] == "log":
            parameters[name] = float(np.exp(np.log(domain["low"]) + value *
                                            (np.log(domain["high"]) - np.log(domain["low"]))))
        else:
            parameters[name] = float(domain["low"] + value * (domain["high"] - domain["low"]))
        if "choices" not in domain:
            # Roundoff in exp/log must not escape a declared domain endpoint.
            parameters[name] = float(np.clip(parameters[name], domain["low"], domain["high"]))
    return parameters


def _canonicalize(points, space):
    points = np.clip(np.asarray(points, dtype=float), 0, 1)
    for column, domain in enumerate(space.values()):
        if "choices" in domain:
            points[:, column] = points[:, column] >= 0.5
    return points


def _propose(observed, scores, space, rng):
    dimensions = len(space)
    kernel = (ConstantKernel(1.0, (1e-3, 1e3)) *
              Matern(np.full(dimensions, 0.3), (1e-2, 10.0), nu=2.5) +
              WhiteKernel(1e-5, (1e-8, 1e-2)))
    gp = GaussianProcessRegressor(kernel=kernel, normalize_y=True, alpha=1e-8,
                                  n_restarts_optimizer=1,
                                  random_state=int(rng.integers(0, 2 ** 31)))
    # Bounds warnings concern surrogate fitting, not failed objective trials.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        gp.fit(observed, scores)
    best_point = observed[int(np.argmax(scores))]
    pool = _canonicalize(np.vstack((rng.random((1024, dimensions)),
                                    best_point + rng.normal(0, 0.08, (512, dimensions)),
                                    best_point + rng.normal(0, 0.25, (512, dimensions)))), space)
    # Count actual evaluations, without rewarding repeats of the same point.
    distances = np.max(np.abs(pool[:, None, :] - observed[None, :, :]), axis=2)
    pool = pool[np.all(distances > 1e-10, axis=1)]
    mean, deviation = gp.predict(pool, return_std=True)
    improvement = mean - max(scores) - 0.001 * max(float(np.std(scores)), 1e-6)
    deviation = np.maximum(deviation, 1e-12)
    z = improvement / deviation
    expected_improvement = improvement * ndtr(z) + deviation * np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)
    return pool[int(np.argmax(expected_improvement))]


def bayesian_optimize(objective, *, base_parameters, n_trials=DEFAULT_N_TRIALS,
                      random_state=42, search_space=None,
                      n_initial_points=DEFAULT_N_INITIAL_POINTS):
    """Maximize ``objective(params)['score']`` with exactly ``n_trials`` calls.

    Initial trials are a warm start plus a Latin hypercube. Later proposals use
    a Matérn Gaussian process and expected improvement. All RNGs are local.
    There is no pruning, accuracy threshold, timeout, or convergence stop.
    A failed objective aborts the comparison instead of giving that method a
    smaller effective budget. Small explicit budgets are supported for smoke
    tests; the normal budget is 12 initial plus 48 adaptive evaluations.
    """
    n_trials = _integer(n_trials, "n_trials")
    n_initial_points = _integer(n_initial_points, "n_initial_points", minimum=2)
    random_state = _integer(random_state, "random_state", minimum=0)
    space = validate_baseline(base_parameters, search_space)
    baseline = _encode(base_parameters, space)
    rng = np.random.default_rng(random_state)
    initial_count = min(n_initial_points, n_trials)
    initial = [baseline]
    if initial_count > 1:
        sampler = qmc.LatinHypercube(d=len(space), seed=int(rng.integers(0, 2 ** 32)))
        initial.extend(_canonicalize(sampler.random(initial_count - 1), space))
    observed, scores, trials = [], [], []
    for index in range(n_trials):
        point = initial[index] if index < initial_count else _propose(
            np.asarray(observed), np.asarray(scores), space, rng)
        params = dict(base_parameters)
        # Preserve the exact supplied baseline, including logarithmic endpoints.
        if index:
            params.update(_decode(point, space))
        metrics = objective(dict(params))
        if not isinstance(metrics, Mapping) or "score" not in metrics:
            raise ValueError("objective must return metrics with a finite score")
        score = metrics["score"]
        if isinstance(score, bool) or not isinstance(score, Real) or not np.isfinite(score):
            raise ValueError("objective score must be a finite real number")
        observed.append(point)
        scores.append(float(score))
        trials.append(dict(metrics, score=float(score), candidate=index, parameters=params,
                           phase="initial" if index < initial_count else "bayesian"))
    selected = int(np.argmax(scores))
    return dict(parameters=dict(trials[selected]["parameters"]), selected_candidate=selected,
                trials=trials, n_trials=n_trials, n_initial_points=initial_count,
                search_space=space)
