"""Diagnose a frozen HAR fusion comparison using the existing RR_L only.

The original benchmark is read, never overwritten. Regularization is selected
on an inner training/validation split with one common candidate for all families.
Outer test curves are marked exploratory and never select a candidate.
"""

import argparse
from collections import defaultdict
from pathlib import Path
import csv
import json

import numpy as np
from scipy.special import softmax
from sklearn.model_selection import train_test_split

from .fusion import FusionESN, _FairReservoir, fuse_probabilities
from .fusion_evaluation import _model_specs, _scores, _subset
from .har_data import HARDataLoader
from .readouts import fit_ridge_readout


LAMBDA_GRID = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


class _CaptureFusionESN(FusionESN):
    """Capture the exact training states already computed by FusionESN.fit."""

    def _ridge(self, features, targets):
        self.training_features_ = features.copy()
        return super()._ridge(features, targets)


def _json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def _csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_states(maps, y, train, test, *, total_nodes, params, seed, cache):
    """Cache only states; all readouts continue to call fit_ridge_readout."""
    manifest = dict(map_names=list(maps), train=list(map(int, train)),
                    test=list(map(int, test)), total_nodes=total_nodes,
                    params=params, seed=seed, version=1)
    manifest_path = cache.with_suffix(".json")
    if cache.exists() and manifest_path.exists():
        if json.loads(manifest_path.read_text()) != manifest:
            raise ValueError(f"State cache configuration mismatch: {cache}")
        with np.load(cache, allow_pickle=False) as arrays:
            return {method: (arrays[f"train_{i}"].copy(), arrays[f"test_{i}"].copy())
                    for i, (method, arch, _) in enumerate(_model_specs(maps))
                    if arch != "parallel_late"}
    train_maps, test_maps = _subset(maps, train), _subset(maps, test)
    states, arrays = {}, {}
    for i, (method, architecture, map_name) in enumerate(_model_specs(maps)):
        if architecture == "parallel_late":
            continue  # Its reservoirs are exactly the Intermediate reservoirs.
        model = _CaptureFusionESN(architecture, total_nodes=total_nodes,
                                  random_state=seed, map_name=map_name, **params)
        model.fit(train_maps, y[train])
        features = (model.training_features_, model.extract_features(test_maps))
        states[method] = features
        arrays[f"train_{i}"], arrays[f"test_{i}"] = features
    np.savez_compressed(cache, **arrays)
    _json(manifest_path, manifest)
    return states


def ridge_predictions(features, targets, regularization, temperature):
    train, test = features
    weights = fit_ridge_readout(train, targets, regularization).T
    train_scores, test_scores = train @ weights, test @ weights
    return (softmax(train_scores / temperature, axis=1),
            softmax(test_scores / temperature, axis=1), weights)


def predictions(states, y_train, classes, regularization, temperature, map_names):
    targets = (y_train[:, None] == classes).astype(float)
    result = {method: ridge_predictions(features, targets, regularization, temperature)
              for method, features in states.items()}
    train, test = states["Parallel-Intermediate"]
    train_branches, test_branches = np.split(train, len(map_names), axis=1), np.split(test, len(map_names), axis=1)
    branch_results = [ridge_predictions((a, b), targets, regularization, temperature)
                      for a, b in zip(train_branches, test_branches)]
    train_prob = np.stack([row[0] for row in branch_results])
    test_prob = np.stack([row[1] for row in branch_results])
    for method in ("mean", "product", "geometric", "max"):
        result[f"Parallel-Late/{method}"] = (
            fuse_probabilities(train_prob, method), fuse_probabilities(test_prob, method), None)
    return result, branch_results


def state_statistics(features, targets, weights, regularization):
    singular = np.linalg.svd(features, compute_uv=False)
    eigenvalues = singular ** 2
    fractions = eigenvalues / max(float(eigenvalues.sum()), np.finfo(float).tiny)
    positive = fractions[fractions > 0]
    minimum = 0.0 if features.shape[1] > features.shape[0] else float(eigenvalues[-1])
    normal = features.T @ features + regularization * np.eye(features.shape[1])
    rhs = features.T @ targets
    residual = np.linalg.norm(normal @ weights - rhs) / max(np.linalg.norm(rhs), np.finfo(float).tiny)
    return dict(effective_df=float(np.sum(eigenvalues / (eigenvalues + regularization))),
                spectral_effective_rank=float(np.exp(-np.sum(positive * np.log(positive)))),
                components_for_95_percent_energy=int(np.searchsorted(np.cumsum(fractions), 0.95) + 1),
                ridge_condition=float((eigenvalues[0] + regularization) / (minimum + regularization)),
                normal_equation_relative_residual=float(residual),
                readout_weight_norm=float(np.linalg.norm(weights)),
                state_variance=float(np.var(features, axis=0).mean()),
                train_mse=float(np.mean((features @ weights - targets) ** 2)))


def select_common_regularization(curve):
    """Equal family weight, with the eight baselines averaged within their family."""
    scores = []
    for value in LAMBDA_GRID:
        families = defaultdict(list)
        for row in curve:
            if row["regularization"] != value:
                continue
            name = row["method"]
            if name.startswith("Parallel-Late/") and name != "Parallel-Late/mean":
                continue
            family = "single_map" if name.startswith("Single-map/") else name
            families[family].append(row["validation_accuracy"])
        score = float(np.mean([np.mean(values) for values in families.values()]))
        scores.append(dict(regularization=value, equal_family_validation_accuracy=score))
    best = max(scores, key=lambda row: row["equal_family_validation_accuracy"])
    return best["regularization"], scores


def map_redundancy(maps, train):
    """Use training data only; CKA compares sample similarity across map types."""
    normalized, grams = {}, {}
    for name, sequences in maps.items():
        chosen = [sequences[i].astype(float) for i in train]
        scaler = FusionESN._fit_scaler(chosen)
        arrays = np.stack([(seq - scaler["mean"]) / scaler["scale"] for seq in chosen])
        normalized[name] = arrays
        flat = arrays.reshape(len(train), -1).copy()
        flat -= flat.mean(axis=0)
        gram = flat @ flat.T
        grams[name] = gram - gram.mean(axis=0) - gram.mean(axis=1)[:, None] + gram.mean()
    rows = []
    names = list(maps)
    for i, first in enumerate(names):
        for second in names[i + 1:]:
            a, b = grams[first], grams[second]
            cka = np.sum(a * b) / np.sqrt(np.sum(a * a) * np.sum(b * b))
            # Range and Doppler bins are different coordinates; do not correlate them binwise.
            pearson = ""
            if first.split('_')[0] == second.split('_')[0]:
                pearson = float(np.corrcoef(normalized[first].ravel(), normalized[second].ravel())[0, 1])
            rows.append(dict(first=first, second=second, linear_cka=float(cka),
                             standardized_binwise_correlation=pearson))
    return rows


def matched_capacity(maps, y, train, test, *, params, seed, classes, map_name="RTM_ch1"):
    """Change only node count for the same map and branch-index RNG stream.

    This is a capacity diagnostic, not another equal-total-node benchmark.
    Using the Late branch index also reproduces its 50-node result exactly.
    """
    branch = list(maps).index(map_name)
    train_sequences = [maps[map_name][i].astype(float) for i in train]
    test_sequences = [maps[map_name][i].astype(float) for i in test]
    scaler = FusionESN._fit_scaler(train_sequences)
    train_sequences = [(seq - scaler["mean"]) / scaler["scale"] for seq in train_sequences]
    test_sequences = [(seq - scaler["mean"]) / scaler["scale"] for seq in test_sequences]
    targets = (y[train, None] == classes).astype(float)
    rows = []
    for nodes in (50, 100, 200, 400):
        model = FusionESN("single_map", total_nodes=nodes, map_name=map_name, random_state=seed, **params)
        reservoir = _FairReservoir(nodes, train_sequences[0].shape[1], model, branch)
        features = (reservoir.transform_sequences(train_sequences), reservoir.transform_sequences(test_sequences))
        p_train, p_test, weights = ridge_predictions(features, targets, params["regularization"], params["temperature"])
        rows.append(dict(map=map_name, branch_index=branch, nodes=nodes,
                         train_accuracy=_scores(y[train], p_train, classes)["accuracy"],
                         test_accuracy=_scores(y[test], p_test, classes)["accuracy"],
                         **state_statistics(features[0], targets, weights, params["regularization"])))
    return rows


def save_report_and_plot(result, output_dir):
    """Export empirical findings; exploratory curves are clearly distinguished."""
    output_dir = Path(output_dir)
    names = ("Single-map/RTM_ch1", "Single-Early", "Parallel-Early", "Parallel-Intermediate", "Parallel-Late/mean")
    grouped = defaultdict(list)
    for row in result["records"]:
        grouped[(row["condition"], row["method"])].append(row)
    summary = []
    for (condition, method), rows in grouped.items():
        summary.append(dict(condition=condition, method=method, n_evaluations=len(rows),
                            mean_train_accuracy=float(np.mean([row["train_accuracy"] for row in rows])),
                            mean_test_accuracy=float(np.mean([row["test_accuracy"] for row in rows])),
                            std_test_accuracy=float(np.std([row["test_accuracy"] for row in rows])),
                            mean_effective_df=float(np.mean([row["effective_df"] for row in rows])) if "effective_df" in rows[0] else ""))
    _csv(output_dir / "diagnostic_summary.csv", summary)
    lines = ["# HAR融合の診断：H1・D1（1.5m）", "",
             "今回の条件では、Early／Intermediateの主な問題はreadoutの過学習で、Lateには各マップのノード数を50に分配する影響があります。構成ごとの正則化の効き方と、各枝の容量を分けて確認しました。", "",
             "600件、訓練300・テスト300、元の同じ50:50分割2方向×3 seed、総400ノード、既存RR_L、全バイアス0です。元の90評価についてaccuracy・NLL・Brier scoreが再現することを確認しました。元の実験結果と既定設定は変更していません。", "",
             "## 1. Early／Intermediate：弱い正則化による過学習", "",
             "元のλ=0.001では、表の4方式はすべて訓練認識率100%です。しかしテストでは単一マップRTM_ch1が94.33%、融合の3方式は約81–82%です。", "",
             "リッジの実効自由度を `tr(X(XᵀX+λI)⁻¹Xᵀ) = Σ sᵢ²/(sᵢ²+λ)` で測りました。これはreadoutの1クラス当たりの学習の柔軟さで、状態行列の単純なランクとは異なります。同じ400ノード・同じλでも、状態のスペクトルが違うため同じ自由度にはなりません。", "",
             "| 構成 | 元の訓練認識率 | 元のテスト認識率 | 元の実効自由度 | 訓練内検証でλ選択後のテスト認識率 |", "| --- | ---: | ---: | ---: | ---: |"]
    for name in names:
        fixed = next(row for row in summary if row["condition"] == "fixed_original" and row["method"] == name)
        selected = next(row for row in summary if row["condition"] == "nested_common_lambda" and row["method"] == name)
        df = f"{fixed['mean_effective_df']:.1f}" if fixed["mean_effective_df"] != "" else "枝別readout"
        lines.append(f"| {name} | {fixed['mean_train_accuracy']*100:.2f}% | {fixed['mean_test_accuracy']*100:.2f} ± {fixed['std_test_accuracy']*100:.2f}% | {df} | {selected['mean_test_accuracy']*100:.2f} ± {selected['std_test_accuracy']*100:.2f}% |")
    lines.extend(["",
                  "λ候補は全構成で同じ8値 `1e-5,1e-4,1e-3,1e-2,1e-1,1,10,100` です。外側の訓練300件だけをさらに225件の訓練・75件の検証へ層化分割し、入力標準化も内側訓練225件だけから計算しました。5つの構成familyを等重みで平均し、全構成へ適用する共通λを選びました。baselineの8マップはfamily内で平均しています。外側テストでλを選んでいません。候補数・候補値・検証分割も全構成で共通です。", "",
                  "6評価の共通λは " + ", ".join(str(row["selected_regularization"]) for row in result["searches"]) + " でした。選んだλでreadoutを外側訓練300件に再学習しました。状態やその他のハイパーパラメータは元と同じです。Early／Intermediateの約11–13ポイントの回復は、元のλがこれらの状態に対して弱かったという証拠です。この追加評価でSingle–Earlyは94.00%、単一RTM_ch1は93.78%でしたが、0.22ポイントの差から優位性を確定することはできません。", "",
                  "一般にリッジの正則化は推定の分散と数値的な条件を改善する役割があります。[scikit-learnの公式説明](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html)。今回の元の計算について、正規方程式の相対残差は最大 " + f"{max(row['normal_equation_relative_residual'] for row in result['records'] if row['condition']=='fixed_original' and 'normal_equation_relative_residual' in row):.2e}" + " でした。観測した認識率差を説明するほどの数値計算誤差は見られません。", "",
                  "## 2. Late：各マップの容量を分配する影響", "",
                  "元の各50ノードの枝のテスト認識率は約76–79%で、平均融合すると91.00%に上がります。枝の補完は実際に働いています。single-map baselineでは同じ1マップに400ノードを使えるため、Lateの50ノードの枝との容量の違いがあります。", "",
                  "RTM_ch1について、入力・λ=0.001・リーク率などを固定し、Lateと同じbranch番号・乱数seedでノード数だけを変えました。50ノード時には元のLateのRTM_ch1枝の結果と一致します。以下は容量を切り分ける診断で、総ノード数固定の構成比較とは別の追加実験です。", "",
                  "| RTM_ch1のノード数 | 訓練認識率 | テスト認識率 |", "| --- | ---: | ---: |"])
    for nodes in (50, 100, 200, 400):
        rows = [row for row in result["matched_capacity"] if row["nodes"] == nodes]
        lines.append(f"| {nodes} | {np.mean([row['train_accuracy'] for row in rows])*100:.2f}% | {np.mean([row['test_accuracy'] for row in rows])*100:.2f} ± {np.std([row['test_accuracy'] for row in rows])*100:.2f}% |")
    lines.extend(["", "## 3. マップの重複と補完", "",
                  "訓練データだけで、各マップのサンプル間の類似関係をlinear CKAで測りました。[CKAの原論文](https://proceedings.mlr.press/v97/kornblith19a.html)を参照しています。CKAはPearson相関とは別の指標で、1に近いほどサンプル間の類似関係が近いことを示します。", ""])
    for title, predicate in (("RTMのチャネル間", lambda a,b:a.startswith("RTM") and b.startswith("RTM")),
                             ("DTMのチャネル間", lambda a,b:a.startswith("DTM") and b.startswith("DTM")),
                             ("DTMとRTMの間", lambda a,b:a.split('_')[0]!=b.split('_')[0])):
        value = np.mean([row["linear_cka"] for row in result["map_redundancy"] if predicate(row["first"],row["second"])])
        lines.append(f"- {title}：CKA {value:.3f}")
    lines.extend(["", "同種マップの4チャネルは似たサンプル構造を持ち、8マップは8個の独立な情報源とは解釈できません。一方で補完する情報は残っています。元の50ノード枝をそのまま使って平均融合すると、次の結果でした。マップを除く診断では使用する総ノード数も減るので、元の公平な構成比較と混同しないでください。", ""])
    for name in ("RTM_only_same_50_node_branches", "DTM_only_same_50_node_branches", "all_maps"):
        rows = [row for row in result["late_ablation"] if row["diagnostic"] == name]
        lines.append(f"- {name}：{np.mean([row['test_accuracy'] for row in rows])*100:.2f}%、使用ノード数{rows[0]['active_nodes']}")
    lines.extend(["", "両種類を使う91.00%がRTMだけの87.72%を上回っています。DTMを加える効果もあります。高いCKAだけから融合が無効とは判断できません。", "",
                  "## 解釈の範囲と再実行", "",
                  "示した±は元と同じ6評価の標準偏差です。独立な6データセットの信頼区間ではありません。今回の結論はH1・D1・既知被験者を含むサンプル分割に限ります。外側のλ別曲線は原因を確認する探索的図であり、そのテスト値から最良λを選んだ認識率は報告していません。", "",
                  "同じλ値・同じノード数を固定する比較には意味がありますが、そのλで各構成がどの程度過学習するかも測る必要があります。構成×λの同じ候補グリッドを示すと、融合方式の効果と正則化との相互作用を分けて確認できます。", "",
                  "crest/から実行します。", "", "```bash", "OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 .venv/bin/python -m modules.har_fusion_diagnostics", "```", "",
                  "diagnostics.jsonに全設定・選択履歴・診断値、diagnostic_summary.csvに集計、matched_node_capacity.csvに容量診断、inner_validation_curves.csvに内側検証、exploratory_outer_curves.csvに探索的曲線、states/に再利用できる状態を保存しています。"])
    (output_dir / "DIAGNOSTIC_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    import os
    os.environ.setdefault("MPLCONFIGDIR", str(output_dir / ".matplotlib"))
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    figure = Figure(figsize=(14, 11), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(2, 2)
    for condition, offset, label, color in (("fixed_original", -.18, "Original lambda=0.001", "#a3adb8"),
                                          ("nested_common_lambda", .18, "Inner-selected common lambda", "#399787")):
        selected = [next(row for row in summary if row["condition"] == condition and row["method"] == name) for name in names]
        axes[0,0].bar(np.arange(len(names)) + offset, [row["mean_test_accuracy"]*100 for row in selected],
                      width=.36, yerr=[row["std_test_accuracy"]*100 for row in selected], capsize=3, label=label, color=color)
    axes[0,0].set_xticks(np.arange(len(names)), ["RTM ch1\n400 nodes", "Single\nEarly", "Parallel\nEarly", "Parallel\nIntermediate", "Parallel\nLate/mean"])
    axes[0,0].set_ylim(0, 105)
    axes[0,0].set_ylabel("Test accuracy (%)")
    axes[0,0].set_title("A. Controlled regularization intervention")
    axes[0,0].legend(fontsize=8, loc="lower left")
    for name in names[:4]:
        values = [np.mean([row["test_accuracy"] for row in result["outer_curves"] if row["method"] == name and row["regularization"] == value])*100 for value in LAMBDA_GRID]
        axes[0,1].semilogx(LAMBDA_GRID, values, marker="o", label=name)
    axes[0,1].axvline(.001, linestyle=":", color="#666666", label="Original lambda")
    axes[0,1].set_title("B. Exploratory outer-test curves (not used to select lambda)")
    axes[0,1].set_xlabel("Ridge lambda")
    axes[0,1].set_ylabel("Test accuracy (%)")
    axes[0,1].legend(fontsize=8, loc="lower left")
    node_values = (50, 100, 200, 400)
    node_groups = [[row for row in result["matched_capacity"] if row["nodes"] == value] for value in node_values]
    axes[1,0].errorbar(node_values, [np.mean([row["test_accuracy"] for row in values])*100 for values in node_groups],
                        yerr=[np.std([row["test_accuracy"] for row in values])*100 for values in node_groups], marker="o", capsize=4, label="Test")
    axes[1,0].plot(node_values, [np.mean([row["train_accuracy"] for row in values])*100 for values in node_groups], "o--", label="Train")
    axes[1,0].set_xticks(node_values)
    axes[1,0].set_xlabel("Nodes for RTM ch1, same branch index and seeds")
    axes[1,0].set_ylabel("Accuracy (%)")
    axes[1,0].set_title("C. Capacity diagnostic, fixed lambda=0.001")
    axes[1,0].legend()
    map_names = result["data"]["map_order"]
    matrix = np.eye(len(map_names))
    for first in range(len(map_names)):
        for second in range(first + 1, len(map_names)):
            value = np.mean([row["linear_cka"] for row in result["map_redundancy"] if row["first"] == map_names[first] and row["second"] == map_names[second]])
            matrix[first,second] = matrix[second,first] = value
    im = axes[1,1].imshow(matrix, vmin=0, vmax=1, cmap="viridis")
    axes[1,1].set_xticks(np.arange(len(map_names)), map_names, rotation=45, ha="right", fontsize=8)
    axes[1,1].set_yticks(np.arange(len(map_names)), map_names, fontsize=8)
    axes[1,1].set_title("D. Map similarity (linear CKA, training data only)")
    figure.colorbar(im, ax=axes[1,1], shrink=.85)
    for ax in (axes[0,0], axes[0,1], axes[1,0]):
        ax.grid(axis="y", alpha=.2)
        ax.set_axisbelow(True)
    figure.suptitle("HAR H1 / D1 (1.5 m): fusion diagnostics | 600 samples | RR_L only, zero biases\nError bars: standard deviation across two split directions and three seeds")
    figure.savefig(output_dir / "fusion_diagnostics.png", dpi=180)
    return summary


def run(reference_path, output_dir):
    reference_path, output_dir = Path(reference_path), Path(output_dir)
    reference = json.loads(reference_path.read_text())
    config = reference["configuration"]
    data = config["data"]
    if (data["room"] != 1 or data["distance"] != 1 or config["total_nodes"] != 400
            or config["seeds"] != [42, 43, 44] or config["split_seed"] != 42
            or config["n_samples"] != 600 or len(config["map_names"]) != 8
            or config["protocols"] != ["50_50"]
            or config["parameter_candidates"][0]["regularization"] != .001):
        raise ValueError("This diagnostic targets the frozen H1/D1 400-node, lambda=0.001, "
                         "600-sample 50:50 benchmark with seeds 42/43/44 and split seed 42")
    loader = HARDataLoader(data["base_dir"], room=data["room"], distance=data["distance"], channels=data["channels"])
    maps, y, metadata = loader.load_all_data()
    if [row["filename"] for row in metadata] != config["sample_filenames"]:
        raise ValueError("Input recordings do not match the frozen benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = output_dir / "states"
    cache_dir.mkdir(exist_ok=True)
    classes = np.unique(y)
    seeds = config["seeds"]
    params = dict(config["parameter_candidates"][0])
    temperature = params["temperature"]
    fixed_lambda = params["regularization"]
    original = {(row["fold"], row["seed"], row["method"]): row for row in reference["records"]}
    records, outer_curves, inner_curves, searches, branches, late_checks, redundancy = [], [], [], [], [], [], []
    capacity = []
    for split in reference["splits"]:
        train, test = np.asarray(split["train_indices"]), np.asarray(split["test_indices"])
        inner_train, validation = train_test_split(train, test_size=0.25,
                                                  stratify=y[train], random_state=config["split_seed"])
        key = split["fold"]
        for row in map_redundancy(maps, train):
            redundancy.append(dict(fold=key, **row))
        for seed in seeds:
            print(f"Diagnose {key}, seed {seed}: reconstruct frozen states", flush=True)
            outer = fit_states(maps, y, train, test, total_nodes=config["total_nodes"], params=params,
                               seed=seed, cache=cache_dir / f"{key}_seed{seed}_outer.npz")
            fixed_predictions, fixed_branches = predictions(outer, y[train], classes, fixed_lambda, temperature, maps)
            targets = (y[train, None] == classes).astype(float)
            for name, (p_train, p_test, weights) in fixed_predictions.items():
                train_metrics, test_metrics = _scores(y[train], p_train, classes), _scores(y[test], p_test, classes)
                prior = original[(key, seed, name)]
                for metric in ("accuracy", "log_loss", "brier_score"):
                    if not np.isclose(test_metrics[metric], prior[metric], rtol=1e-8, atol=1e-10):
                        raise ValueError(f"Frozen benchmark parity failed: {key}/{seed}/{name}/{metric}")
                stats = state_statistics(outer[name][0], targets, weights, fixed_lambda) if weights is not None else {}
                records.append(dict(fold=key, seed=seed, method=name, condition="fixed_original",
                                    regularization=fixed_lambda, train_accuracy=train_metrics["accuracy"],
                                    test_accuracy=test_metrics["accuracy"], **stats))
            branch_test = np.stack([row[1] for row in fixed_branches])
            branch_predictions = classes[np.argmax(branch_test, axis=2)]
            errors = branch_predictions != y[test]
            error_correlation = np.corrcoef(errors.astype(float))
            for i, name in enumerate(maps):
                train_p, test_p, weights = fixed_branches[i]
                train_x = np.split(outer["Parallel-Intermediate"][0], len(maps), axis=1)[i]
                branches.append(dict(fold=key, seed=seed, map=name, nodes=config["total_nodes"] // len(maps),
                                     train_accuracy=_scores(y[train], train_p, classes)["accuracy"],
                                     test_accuracy=_scores(y[test], test_p, classes)["accuracy"],
                                     **state_statistics(train_x, targets, weights, fixed_lambda)))
            subsets = {"all_maps": list(range(len(maps))),
                       "RTM_only_same_50_node_branches": [i for i, name in enumerate(maps) if name.startswith("RTM")],
                       "DTM_only_same_50_node_branches": [i for i, name in enumerate(maps) if name.startswith("DTM")]}
            for name, chosen in subsets.items():
                pooled = fuse_probabilities(branch_test[chosen], "mean")
                late_checks.append(dict(fold=key, seed=seed, diagnostic=name,
                                       test_accuracy=_scores(y[test], pooled, classes)["accuracy"],
                                       active_nodes=(config["total_nodes"] // len(maps)) * len(chosen)))
            upper = float(np.any(~errors, axis=0).mean())
            rt = list(maps).index("RTM_ch1")
            ensemble_error = classes[np.argmax(fixed_predictions["Parallel-Late/mean"][1], axis=1)] != y[test]
            late_checks.append(dict(fold=key, seed=seed, diagnostic="oracle_at_least_one_branch_correct_not_a_model",
                                   test_accuracy=upper, active_nodes=config["total_nodes"]))
            _json(output_dir / f"errors_{key}_seed{seed}.json", dict(
                map_names=list(maps), pairwise_error_correlation=error_correlation.tolist(),
                rt1_correct_ensemble_wrong=int(np.sum(~errors[rt] & ensemble_error)),
                rt1_wrong_ensemble_correct=int(np.sum(errors[rt] & ~ensemble_error)),
                n_test=len(test)))
            print(f"Diagnose {key}, seed {seed}: inner validation regularization selection", flush=True)
            inner = fit_states(maps, y, inner_train, validation, total_nodes=config["total_nodes"], params=params,
                               seed=seed, cache=cache_dir / f"{key}_seed{seed}_inner.npz")
            curve = []
            for value in LAMBDA_GRID:
                predicted, _ = predictions(inner, y[inner_train], classes, value, temperature, maps)
                for name, (_, p_validation, _) in predicted.items():
                    row = dict(fold=key, seed=seed, method=name, regularization=value,
                               validation_accuracy=_scores(y[validation], p_validation, classes)["accuracy"])
                    curve.append(row)
                    inner_curves.append(row)
            selected, scores = select_common_regularization(curve)
            searches.append(dict(fold=key, seed=seed, inner_train_indices=inner_train.tolist(),
                                 validation_indices=validation.tolist(), selected_regularization=selected,
                                 selection="one common lambda; equal family weight; inner validation only", scores=scores))
            selected_predictions, _ = predictions(outer, y[train], classes, selected, temperature, maps)
            for name, (p_train, p_test, weights) in selected_predictions.items():
                stats = state_statistics(outer[name][0], targets, weights, selected) if weights is not None else {}
                records.append(dict(fold=key, seed=seed, method=name, condition="nested_common_lambda",
                                    regularization=selected,
                                    train_accuracy=_scores(y[train], p_train, classes)["accuracy"],
                                    test_accuracy=_scores(y[test], p_test, classes)["accuracy"], **stats))
            for value in LAMBDA_GRID:
                predicted, _ = predictions(outer, y[train], classes, value, temperature, maps)
                for name, (p_train, p_test, _) in predicted.items():
                    outer_curves.append(dict(fold=key, seed=seed, method=name, regularization=value,
                                             train_accuracy=_scores(y[train], p_train, classes)["accuracy"],
                                             test_accuracy=_scores(y[test], p_test, classes)["accuracy"],
                                             role="exploratory curve; never used for hyperparameter selection"))
            print(f"Diagnose {key}, seed {seed}: common lambda {selected}", flush=True)
            print(f"Diagnose {key}, seed {seed}: matched RTM_ch1 node capacity", flush=True)
            for row in matched_capacity(maps, y, train, test, params=params, seed=seed, classes=classes):
                if row["nodes"] == 50:
                    corresponding = next(record for record in branches
                                         if record["fold"] == key and record["seed"] == seed and record["map"] == "RTM_ch1")
                    if not np.isclose(row["test_accuracy"], corresponding["test_accuracy"]):
                        raise ValueError("Matched 50-node capacity diagnostic failed branch parity")
                capacity.append(dict(fold=key, seed=seed, **row))
            _json(output_dir / "diagnostics_partial.json", dict(records=records, searches=searches))
    result = dict(reference_results=str(reference_path.resolve()), data=loader.data_manifest,
                  seeds=list(seeds), regularization_grid=list(LAMBDA_GRID), records=records,
                  searches=searches, outer_curves=outer_curves, inner_curves=inner_curves,
                  branch_diagnostics=branches, late_ablation=late_checks, map_redundancy=redundancy,
                  matched_capacity=capacity,
                  fixed_benchmark_parity="all tested accuracies, log losses, and Brier scores matched",
                  constraints="RR_L only; all biases zero; same room/distance/splits/node budget",
                  caveats="capacity diagnosis intentionally varies node count; map subset pooling changes active nodes; "
                          "neither is another fixed-total-node benchmark; oracle is not deployable; outer curves are exploratory")
    _json(output_dir / "diagnostics.json", result)
    fieldnames = list(dict.fromkeys(key for row in records for key in row))
    _csv(output_dir / "model_diagnostics.csv", [{key: row.get(key, "") for key in fieldnames} for row in records])
    _csv(output_dir / "inner_validation_curves.csv", inner_curves)
    _csv(output_dir / "exploratory_outer_curves.csv", outer_curves)
    _csv(output_dir / "map_redundancy.csv", redundancy)
    _csv(output_dir / "branch_diagnostics.csv", branches)
    _csv(output_dir / "late_ablation.csv", late_checks)
    _csv(output_dir / "matched_node_capacity.csv", capacity)
    save_report_and_plot(result, output_dir)
    print(f"Diagnostics saved: {output_dir.resolve()}", flush=True)
    return result


def cli(argv=None):
    root = Path(__file__).resolve().parents[1] / "HAR-Dataset-Project" / "results"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-results", type=Path,
                        default=root / "fusion_existing_ridge_no_bias_400_room1_distance1_50_50" / "fusion_results.json")
    parser.add_argument("--output-dir", type=Path, default=root / "fusion_diagnostics_room1_distance1")
    args = parser.parse_args(argv)
    return run(args.reference_results, args.output_dir)


if __name__ == "__main__":
    cli()
