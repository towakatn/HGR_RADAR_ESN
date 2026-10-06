"""Paired, leakage-free evaluation of reservoir topology and feature fusion."""

from collections import defaultdict
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
import csv
import json
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedKFold, train_test_split

from .fusion import FusionESN, LATE_FUSION_METHODS, fuse_probabilities


PROTOCOLS = ("50_50", "10fold", "session_split", "loso")
RESERVOIR_DEFAULTS = dict(spectral_radius=0.95, input_scaling=0.2, density=0.1,
                         leakage_rate=0.05, bias_scaling=0.0,
                         temperature=1.0, standardize_inputs=True)
CANDIDATE_KEYS = frozenset((*RESERVOIR_DEFAULTS, "regularization"))


@dataclass(frozen=True)
class EvaluationSplit:
    protocol: str
    fold: str
    train_indices: np.ndarray
    test_indices: np.ndarray


def build_soli_maps(X_md, X_rtm, channels=None):
    """Expose DTM/RTM channel maps with one consistent, interleaved order."""
    channels = tuple(X_md) if channels is None else tuple(channels)
    if not channels or len(set(channels)) != len(channels):
        raise ValueError("channels must be nonempty and unique")
    if set(X_md) != set(channels) or set(X_rtm) != set(channels):
        raise ValueError("DTM and RTM must contain exactly the requested channels")
    return {name: sequences for ch in channels
            for name, sequences in ((f"DTM_ch{ch}", X_md[ch]), (f"RTM_ch{ch}", X_rtm[ch]))}


def _validate_labels(y, metadata):
    y = np.asarray(y)
    if y.ndim != 1 or not len(y) or len(metadata) != len(y):
        raise ValueError("y and metadata must contain the same nonzero number of samples")
    if len(np.unique(y)) < 2:
        raise ValueError("comparison requires at least two classes")
    return y


def _split(protocol, fold, train, test):
    train, test = np.asarray(train, dtype=int), np.asarray(test, dtype=int)
    if not len(train) or not len(test) or np.intersect1d(train, test).size:
        raise ValueError(f"{protocol}/{fold} needs nonempty, disjoint train/test sets")
    return EvaluationSplit(protocol, str(fold), train, test)


def make_soli_splits(y, metadata, *, protocols=PROTOCOLS, n_splits=10,
                     split_seed=42, test_size=0.5):
    """Create outer splits once, shared by all methods and reservoir seeds.

    The session protocol retains the existing loader's interpretation of the
    filename sample index as a session identifier. It must be verified against
    acquisition metadata before being described as separate recording sessions.
    """
    y = _validate_labels(y, metadata)
    protocols = tuple(protocols)
    if not protocols or len(set(protocols)) != len(protocols) or set(protocols) - set(PROTOCOLS):
        raise ValueError(f"protocols must be unique members of {PROTOCOLS}")
    if not isinstance(split_seed, Integral) or isinstance(split_seed, bool) or split_seed < 0:
        raise ValueError("split_seed must be a nonnegative integer")
    indices, splits = np.arange(len(y)), []
    for protocol in protocols:
        if protocol == "50_50":
            train, test = train_test_split(indices, test_size=test_size, stratify=y,
                                           random_state=int(split_seed))
            splits.extend((_split(protocol, "pattern1", train, test),
                           _split(protocol, "pattern2", test, train)))
        elif protocol == "10fold":
            if not isinstance(n_splits, Integral) or isinstance(n_splits, bool) or n_splits < 2:
                raise ValueError("n_splits must be an integer >= 2")
            if np.min(np.unique(y, return_counts=True)[1]) < n_splits:
                raise ValueError("each class needs at least n_splits samples")
            cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=int(split_seed))
            splits.extend(_split(protocol, fold, train, test)
                          for fold, (train, test) in enumerate(cv.split(indices, y)))
        else:
            try:
                subjects = np.asarray([row["subject"] for row in metadata])
                sessions = np.asarray([row["session"] for row in metadata]) if protocol == "session_split" else None
            except KeyError as exc:
                raise ValueError(f"{protocol} requires subject/session metadata") from exc
            unique_subjects = np.unique(subjects)
            if protocol == "loso" and len(unique_subjects) < 2:
                raise ValueError("LOSO requires at least two subjects")
            for subject in unique_subjects:
                subject_indices = indices[subjects == subject]
                if protocol == "loso":
                    splits.append(_split(protocol, subject, indices[subjects != subject], subject_indices))
                    continue
                subject_sessions = sessions[subjects == subject]
                unique_sessions = np.unique(subject_sessions)
                if len(unique_sessions) < 2:
                    raise ValueError(f"subject {subject} needs at least two session identifiers")
                subject_id = (int(subject) if isinstance(subject, Integral)
                              else sum(ord(char) for char in str(subject)))
                rng = np.random.RandomState((int(split_seed) + subject_id) % (2 ** 32))
                shuffled = rng.permutation(unique_sessions)
                train_mask = np.isin(subject_sessions, shuffled[:len(shuffled) // 2])
                splits.append(_split(protocol, subject, subject_indices[train_mask], subject_indices[~train_mask]))
    return splits


def _subset(maps, indices):
    return {name: [sequences[i] for i in indices] for name, sequences in maps.items()}


def _model_specs(map_names):
    return [(f"Single-map/{name}", "single_map", name) for name in map_names] + [
        ("Single-Early", "single_early", None),
        ("Parallel-Early", "parallel_early", None),
        ("Parallel-Intermediate", "parallel_intermediate", None),
        ("Parallel-Late", "parallel_late", None),
    ]


def _scores(y, probabilities, classes):
    predictions = classes[np.argmax(probabilities, axis=1)]
    class_columns = {label: index for index, label in enumerate(classes)}
    true_columns = np.asarray([class_columns[label] for label in y])
    truth = np.eye(len(classes))[true_columns]
    return dict(accuracy=float(np.mean(predictions == y)),
                balanced_accuracy=float(balanced_accuracy_score(y, predictions)),
                log_loss=float(-np.log(np.clip(probabilities[np.arange(len(y)), true_columns],
                                              np.finfo(float).eps, 1.0)).mean()),
                brier_score=float(np.sum((probabilities - truth) ** 2, axis=1).mean()))


def _align_probabilities(probabilities, model_classes, classes):
    aligned = np.zeros((len(probabilities), len(classes)))
    columns = {label: index for index, label in enumerate(classes)}
    for index, label in enumerate(model_classes):
        aligned[:, columns[label]] = probabilities[:, index]
    return aligned


def _evaluate_methods(train_maps, test_maps, y_train, y_test, classes, *,
                      total_nodes, params, seed, late_methods):
    results = []
    for method, architecture, map_name in _model_specs(train_maps):
        model = FusionESN(architecture, total_nodes=total_nodes, random_state=seed,
                          map_name=map_name, **params)
        start = time.perf_counter()
        model.fit(train_maps, y_train)
        fit_seconds = time.perf_counter() - start
        start = time.perf_counter()
        features = model.extract_features(test_maps)
        feature_seconds = time.perf_counter() - start
        common = dict(architecture=architecture, map_name=map_name,
                      total_nodes=sum(model.node_counts_), nodes_per_reservoir=list(model.node_counts_),
                      regularization=model.regularization, parameter_counts=model.parameter_counts,
                      fit_seconds=fit_seconds, test_feature_seconds=feature_seconds)
        if architecture == "parallel_late":
            start = time.perf_counter()
            branches = model.branch_probabilities_from_features(features)
            branch_seconds = time.perf_counter() - start
            for fusion in late_methods:
                start = time.perf_counter()
                probabilities = fuse_probabilities(branches, fusion)
                predict_seconds = branch_seconds + time.perf_counter() - start
                aligned = _align_probabilities(probabilities, model.classes_, classes)
                results.append(dict(common, method=f"{method}/{fusion}", late_fusion=fusion,
                                    predict_seconds=predict_seconds, **_scores(y_test, aligned, classes)))
        else:
            start = time.perf_counter()
            probabilities = model.predict_proba_from_features(features)
            predict_seconds = time.perf_counter() - start
            aligned = _align_probabilities(probabilities, model.classes_, classes)
            results.append(dict(common, method=method, late_fusion=None,
                                predict_seconds=predict_seconds, **_scores(y_test, aligned, classes)))
    return results


def _inner_split(y, metadata, outer, seed, validation_size):
    """Use only outer training samples, retaining group separation when applicable."""
    train = outer.train_indices
    if outer.protocol in ("loso", "session_split"):
        key = "subject" if outer.protocol == "loso" else "session"
        groups = np.asarray([metadata[i][key] for i in train])
        if len(np.unique(groups)) < 2:
            raise ValueError(f"{outer.protocol} needs >= 2 training {key} groups for nested search")
        splitter = GroupShuffleSplit(n_splits=1, test_size=validation_size, random_state=seed)
        inner_train, validation = next(splitter.split(train, y[train], groups))
    else:
        inner_train, validation = train_test_split(np.arange(len(train)), test_size=validation_size,
                                                  stratify=y[train], random_state=seed)
    return train[inner_train], train[validation]


def _select_common_params(maps, y, metadata, classes, outer, candidates, total_nodes,
                          seed, inner_validation_size, inner_split_seed):
    train, validation = _inner_split(y, metadata, outer, inner_split_seed, inner_validation_size)
    if len(np.unique(y[train])) < 2:
        raise ValueError("inner training split needs at least two classes")
    train_maps, validation_maps = _subset(maps, train), _subset(maps, validation)
    trials = []
    for index, params in enumerate(candidates):
        rows = _evaluate_methods(train_maps, validation_maps, y[train], y[validation], classes,
                                 total_nodes=total_nodes, params=params, seed=seed, late_methods=("mean",))
        families = defaultdict(list)
        for row in rows:
            families[row["architecture"]].append(row["accuracy"])
        score = float(np.mean([np.mean(values) for values in families.values()]))
        trials.append(dict(candidate=index, parameters=params, family_mean_accuracy=score,
                           method_accuracies={row["method"]: row["accuracy"] for row in rows}))
    best = max(range(len(trials)), key=lambda index: trials[index]["family_mean_accuracy"])
    return best, dict(protocol=outer.protocol, fold=outer.fold, seed=seed,
                      inner_split_seed=inner_split_seed,
                      train_indices=train.tolist(), validation_indices=validation.tolist(),
                      selected_candidate=best, trials=trials)


def _summarize(records):
    grouped = defaultdict(list)
    for record in records:
        grouped[(record["protocol"], record["method"])].append(record)
    summary = []
    for (protocol, method), rows in grouped.items():
        seed_means = {seed: float(np.mean([row["accuracy"] for row in rows if row["seed"] == seed]))
                      for seed in sorted({row["seed"] for row in rows})}
        result = dict(protocol=protocol, method=method, total_nodes=rows[0]["total_nodes"],
                      n_seeds=len(seed_means), n_evaluations=len(rows), seed_mean_accuracy=seed_means)
        for metric in ("accuracy", "balanced_accuracy", "log_loss", "brier_score"):
            values = [row[metric] for row in rows]
            result[f"mean_{metric}"] = float(np.mean(values))
            result[f"std_{metric}"] = float(np.std(values))
        result["std_seed_mean_accuracy"] = float(np.std(list(seed_means.values())))
        summary.append(result)
    return summary


def _contrasts(records):
    paired = defaultdict(dict)
    for row in records:
        paired[(row["protocol"], row["fold"], row["seed"])][row["method"]] = row["accuracy"]
    comparisons = (
        ("parallel_topology_with_early_fusion", "Parallel-Early", "Single-Early"),
        ("map_specific_inputs_with_parallel_reservoirs", "Parallel-Intermediate", "Parallel-Early"),
        ("late_vs_intermediate_readout", "Parallel-Late/mean", "Parallel-Intermediate"),
    )
    by_comparison = defaultdict(list)
    for (protocol, fold, seed), accuracies in paired.items():
        for effect, target, reference in comparisons:
            if target in accuracies and reference in accuracies:
                by_comparison[(protocol, effect, target, reference)].append(accuracies[target] - accuracies[reference])
    return [dict(protocol=key[0], effect=key[1], target=key[2], reference=key[3],
                 mean_accuracy_difference=float(np.mean(values)),
                 std_accuracy_difference=float(np.std(values)), n_pairs=len(values))
            for key, values in by_comparison.items()]


def run_fusion_comparison(maps, y, metadata, *, total_nodes=400, regularization=0.001,
                          reservoir_params=None, seeds=(42, 43, 44), protocols=PROTOCOLS,
                          n_splits=10, split_seed=42, n_trials=1, parameter_candidates=None,
                          late_fusion_methods=("mean", "product", "geometric", "max"),
                          inner_validation_size=0.25, progress=True):
    """Compare all methods on paired outer folds using shared parameters.

    One trial fixes the supplied parameters and performs no search. Multiple
    trials evaluate the same candidate sequence on one common inner split, then
    select ONE setting for all methods. The selection score averages the five
    architecture families equally; individual single-map scores are averaged
    inside their family. Outer test labels never select hyperparameters.
    """
    y = _validate_labels(y, metadata)
    if not maps or any(len(sequences) != len(y) for sequences in maps.values()):
        raise ValueError("all maps must align with y and metadata")
    if not isinstance(total_nodes, Integral) or isinstance(total_nodes, bool) or total_nodes < len(maps) or total_nodes % len(maps):
        raise ValueError("total_nodes must be >= map count and divisible by map count")
    seeds = tuple(seeds)
    if not seeds or len(set(seeds)) != len(seeds) or any(isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0 for seed in seeds):
        raise ValueError("seeds must be unique nonnegative integers")
    seeds = tuple(int(seed) for seed in seeds)
    protocols = tuple(protocols)
    if not isinstance(n_trials, Integral) or isinstance(n_trials, bool) or n_trials < 1:
        raise ValueError("n_trials must be a positive integer")
    if not 0 < inner_validation_size < 1:
        raise ValueError("inner_validation_size must be between zero and one")
    late_methods = tuple(late_fusion_methods)
    if not late_methods or len(set(late_methods)) != len(late_methods) or set(late_methods) - set(LATE_FUSION_METHODS):
        raise ValueError(f"late_fusion_methods must be unique members of {LATE_FUSION_METHODS}")
    base = dict(RESERVOIR_DEFAULTS, regularization=regularization)
    if reservoir_params is not None:
        if set(reservoir_params) - set(RESERVOIR_DEFAULTS):
            raise ValueError("unsupported reservoir_params key")
        base.update(reservoir_params)
    if parameter_candidates is None:
        if n_trials != 1:
            raise ValueError("n_trials > 1 requires exactly n_trials shared parameter_candidates")
        parameter_candidates = [{}]
    if len(parameter_candidates) != n_trials:
        raise ValueError("parameter_candidates length must equal n_trials")
    candidates = []
    for candidate in parameter_candidates:
        if not isinstance(candidate, dict) or set(candidate) - CANDIDATE_KEYS:
            raise ValueError(f"candidate keys must be among {sorted(CANDIDATE_KEYS)}")
        params = dict(base, **candidate)
        # Validate parameters without initializing any reservoir or consuming RNG.
        validated = FusionESN("single_early", total_nodes=total_nodes, **params)
        params = {key: getattr(validated, key) for key in params}
        candidates.append(params)
    outer_splits = make_soli_splits(y, metadata, protocols=protocols, n_splits=n_splits,
                                   split_seed=split_seed)
    classes = np.unique(y)
    records, searches = [], []
    start = time.perf_counter()
    for split_index, outer in enumerate(outer_splits, start=1):
        train_maps, test_maps = _subset(maps, outer.train_indices), _subset(maps, outer.test_indices)
        for seed in seeds:
            if progress:
                print(f"Fusion {split_index}/{len(outer_splits)}: {outer.protocol}/{outer.fold}, seed={seed}", flush=True)
            selected = 0
            if n_trials > 1:
                selected, search = _select_common_params(maps, y, metadata, classes, outer, candidates,
                                                         total_nodes, seed, inner_validation_size, int(split_seed))
                searches.append(search)
            rows = _evaluate_methods(train_maps, test_maps, y[outer.train_indices], y[outer.test_indices], classes,
                                     total_nodes=total_nodes, params=candidates[selected], seed=seed,
                                     late_methods=late_methods)
            records.extend(dict(row, protocol=outer.protocol, fold=outer.fold, seed=seed,
                                n_train=len(outer.train_indices), n_test=len(outer.test_indices),
                                n_search_trials=n_trials if n_trials > 1 else 0,
                                selected_candidate=selected, parameters=candidates[selected]) for row in rows)
    return dict(configuration=dict(total_nodes=int(total_nodes), map_names=list(maps),
                                   seeds=list(seeds), split_seed=int(split_seed), n_splits=int(n_splits),
                                   protocols=list(protocols), n_trials=int(n_trials), parameter_candidates=candidates,
                                   late_fusion_methods=list(late_methods), inner_validation_size=inner_validation_size,
                                   n_samples=len(y), classes=classes.tolist(),
                                   readout="existing_RR_L",
                                   readout_implementation="modules.readouts.fit_ridge_readout",
                                   readout_objective="sum_sample_squared_error + lambda * squared_weight_norm",
                                   search_selection="one shared candidate; equally weighted architecture families",
                                   session_identifier="filename third integer, as interpreted by existing loader"),
                elapsed_seconds=time.perf_counter() - start,
                splits=[dict(protocol=split.protocol, fold=split.fold,
                             train_indices=split.train_indices.tolist(), test_indices=split.test_indices.tolist())
                        for split in outer_splits], search=searches, records=records,
                summary=_summarize(records), contrasts=_contrasts(records))


def save_fusion_results(results, output_dir):
    """Persist raw paired measurements, summaries, contrasts, and split indices."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "fusion_results.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, ensure_ascii=False, indent=2, allow_nan=False, default=_json_default)
    for key, filename in (("summary", "fusion_summary.csv"), ("records", "fusion_records.csv"),
                          ("contrasts", "fusion_contrasts.csv")):
        rows = results[key]
        if not rows:
            continue
        with (output_dir / filename).open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({name: json.dumps(value, ensure_ascii=False, default=_json_default) if isinstance(value, (dict, list)) else value
                                 for name, value in row.items()})
    return output_dir


def _json_default(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Unsupported report value: {type(value).__name__}")


def save_fusion_plot(results, output_dir):
    """Export a standalone scientific plot; bars show paired-fold/seed dispersion."""
    import os
    import math

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    protocols = list(dict.fromkeys(row["protocol"] for row in results["summary"]))
    columns = min(2, len(protocols))
    figure = Figure(figsize=(9 * columns, 7.5 * math.ceil(len(protocols) / columns)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = np.atleast_1d(figure.subplots(math.ceil(len(protocols) / columns), columns)).ravel()
    for axis, protocol in zip(axes, protocols):
        rows = [row for row in results["summary"] if row["protocol"] == protocol]
        names = [row["method"] for row in rows]
        means = np.asarray([row["mean_accuracy"] * 100 for row in rows])
        deviations = np.asarray([row["std_accuracy"] * 100 for row in rows])
        colors = ["#a3adb8" if name.startswith("Single-map/") else
                  "#8f6cb3" if name == "Single-Early" else
                  "#de9745" if name == "Parallel-Early" else
                  "#399787" if name == "Parallel-Intermediate" else "#5082b5" for name in names]
        axis.barh(np.arange(len(rows)), means, xerr=deviations, color=colors,
                  error_kw=dict(elinewidth=1, capsize=3), height=0.68)
        axis.set_yticks(np.arange(len(rows)), names)
        axis.invert_yaxis()
        axis.set_xlim(0, 108)
        axis.set_xticks(np.arange(0, 101, 20))
        axis.set_xlabel("Accuracy (%) | error bars: standard deviation across folds and seeds")
        axis.set_title(protocol)
        axis.grid(axis="x", alpha=0.2)
        axis.set_axisbelow(True)
        for index, value in enumerate(means):
            axis.text(min(value + deviations[index] + 1, 102), index, f"{value:.2f}", va="center", fontsize=9)
    for axis in axes[len(protocols):]:
        axis.set_visible(False)
    config = results["configuration"]
    readout_title = f"\nReadout: {config['readout']}" if "readout" in config else ""
    dataset = config.get('dataset', 'Soli')
    data = config.get('data', {})
    conditions = []
    if 'room' in data:
        conditions.append(f"Room H{data['room']}")
    if 'distance' in data:
        distance_label = f"Distance D{data['distance']}"
        if 'distance_meters' in data:
            distance_label += f" ({data['distance_meters']} m)"
        conditions.append(distance_label)
    condition_title = '\n' + ' | '.join(conditions) if conditions else ''
    figure.suptitle(f"{dataset} reservoir / fusion comparison | N={config['total_nodes']} | "
                    f"{config['n_samples']} samples | seeds={config['seeds']}{condition_title}{readout_title}")
    destination = output_dir / "fusion_summary.png"
    figure.savefig(destination, dpi=180)
    return destination
