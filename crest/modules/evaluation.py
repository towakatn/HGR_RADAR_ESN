#!/usr/bin/env python3
"""
共通評価モジュール
全リードアウト手法で使用する評価関数を提供

精度を維持するため、元ファイル(rc_10fold_cross_validation_rf_fast.py)と
完全に同一のシード・分割ロジックを使用:
  - 50:50分割: train_test_split(random_state=reservoir_random_state+1)
  - 10-Fold CV: StratifiedKFold(random_state=reservoir_random_state+1)
    * fold毎: 接続行列seed=reservoir_random_state+fold_num
    * 分類器seed=reservoir_random_state+fold_num
  - session分割: np.random.seed(42 + subj_id)
  - LOSO/session/50:50の分類器: random_state=reservoir_random_state
"""

import time
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

from .reservoir_computer import ReservoirComputer


def _compute_all_states(X, reservoir_config):
    """
    全サンプルのリザバー状態を一括計算
    50:50, LOSO, session分割で使用（全て同一接続行列のため）
    """
    rc = ReservoirComputer(
        n_reservoir=reservoir_config['n_reservoir'],
        spectral_radius=reservoir_config['spectral_radius'],
        input_scaling=reservoir_config['input_scaling'],
        density=reservoir_config['density'],
        leakage_rate=reservoir_config['leakage_rate'],
        random_state=reservoir_config['random_state']
    )
    t0 = time.time()
    rc.fit(X, np.zeros(len(X)))
    elapsed = time.time() - t0
    print(f"({elapsed:.1f}秒)", end=' ', flush=True)
    return rc.states


def _compute_fold_states(X_train, X_test, reservoir_config, fold_num):
    """
    10-Fold CV用: fold毎に異なる接続行列でリザバー状態を計算
    元コードのevaluate_single_foldと完全に同一のシードロジック
    """
    print(f"    Fold {fold_num+1}/10:", end=' ', flush=True)
    rs = reservoir_config['random_state']
    fold_base_seed = rs + fold_num
    conn_seed = int(fold_base_seed)
    train_seed = int(fold_base_seed + 1000)
    test_seed = int(fold_base_seed + 2000)

    n_inputs = X_train[0].shape[1]

    # 接続行列を生成（fold毎に異なるseed）
    print("接続生成...", end=' ', flush=True)
    temp_rc = ReservoirComputer(
        n_reservoir=reservoir_config['n_reservoir'],
        spectral_radius=reservoir_config['spectral_radius'],
        input_scaling=reservoir_config['input_scaling'],
        density=reservoir_config['density'],
        leakage_rate=reservoir_config['leakage_rate'],
        random_state=conn_seed
    )
    temp_rc._initialize_reservoir(n_inputs)
    W_res = temp_rc.W_reservoir.copy()
    W_in = temp_rc.W_input.copy()

    # 訓練データの状態計算（接続行列を注入）
    rc_train = ReservoirComputer(
        n_reservoir=reservoir_config['n_reservoir'],
        spectral_radius=reservoir_config['spectral_radius'],
        input_scaling=reservoir_config['input_scaling'],
        density=reservoir_config['density'],
        leakage_rate=reservoir_config['leakage_rate'],
        random_state=train_seed
    )
    rc_train.W_reservoir = W_res.copy()
    rc_train.W_input = W_in.copy()
    print(f"訓練({len(X_train)}サンプル)...", end=' ', flush=True)
    rc_train.fit(X_train, np.zeros(len(X_train)))
    train_states = rc_train.states

    # テストデータの状態計算（同一接続行列を注入）
    print(f"テスト({len(X_test)}サンプル)...", end=' ', flush=True)
    rc_test = ReservoirComputer(
        n_reservoir=reservoir_config['n_reservoir'],
        spectral_radius=reservoir_config['spectral_radius'],
        input_scaling=reservoir_config['input_scaling'],
        density=reservoir_config['density'],
        leakage_rate=reservoir_config['leakage_rate'],
        random_state=test_seed
    )
    rc_test.W_reservoir = W_res.copy()
    rc_test.W_input = W_in.copy()
    rc_test.fit(X_test, np.zeros(len(X_test)))
    test_states = rc_test.states
    print("✓")

    return train_states, test_states


def evaluate_50_50(all_states, y, reservoir_config, classifiers):
    """
    50:50データ分割評価（1方向のみ: 元コードと同一）

    Args:
        all_states: 事前計算済みリザバー状態(n_samples, n_reservoir)
        y: ラベル配列
        reservoir_config: リザバー設定
        classifiers: [(name, create_fn)] のリスト

    Returns:
        dict: {clf_name: accuracy}
    """
    rs = reservoir_config['random_state']
    indices = np.arange(len(y))
    train_idx, test_idx = train_test_split(
        indices, test_size=0.5, stratify=y,
        random_state=(rs + 1) if rs is not None else None
    )

    train_states = all_states[train_idx]
    test_states = all_states[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    results = {}
    for name, create_fn in classifiers:
        clf = create_fn(rs)
        clf.fit(train_states, y_train)
        y_pred = clf.predict(test_states)
        results[name] = accuracy_score(y_test, y_pred)

    return results


def evaluate_10fold(X, y, reservoir_config, classifiers, n_splits=10):
    """
    10分割交差検証
    fold毎に異なる接続行列を使用（元コードと同一）

    Args:
        X: 入力データリスト（可変長信号）
        y: ラベル配列
        reservoir_config: リザバー設定
        classifiers: [(name, create_fn)] のリスト
        n_splits: 分割数

    Returns:
        dict: {clf_name: {'mean_accuracy': float, 'std_accuracy': float}}
    """
    rs = reservoir_config['random_state']
    skf = StratifiedKFold(
        n_splits=n_splits, shuffle=True,
        random_state=(rs + 1) if rs is not None else None
    )

    fold_accs = {name: [] for name, _ in classifiers}

    print()  # 改行
    for fold_num, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train = [X[i] for i in train_idx]
        X_test = [X[i] for i in test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # fold毎の接続行列でリザバー状態を計算
        train_states, test_states = _compute_fold_states(
            X_train, X_test, reservoir_config, fold_num
        )

        # 分類器のseed: 元コードのfold_random_state = random_state + fold_num
        fold_rs = rs + fold_num if rs is not None else None

        for name, create_fn in classifiers:
            clf = create_fn(fold_rs)
            clf.fit(train_states, y_train)
            y_pred = clf.predict(test_states)
            fold_accs[name].append(accuracy_score(y_test, y_pred))

    results = {}
    for name, _ in classifiers:
        accs = fold_accs[name]
        results[name] = {
            'mean_accuracy': np.mean(accs),
            'std_accuracy': np.std(accs),
        }

    return results


def evaluate_session_split(all_states, y, metadata, reservoir_config, classifiers):
    """
    被験者内セッション50:50分割（元コードのevaluate_leave_one_session_outと同一）

    セッション分割のseed: np.random.seed(42 + subj_id)
    subj_id = sum(ord(c) for c in str(subject)) % 10000

    Args:
        all_states: 事前計算済みリザバー状態
        y: ラベル配列
        metadata: メタデータリスト
        reservoir_config: リザバー設定
        classifiers: [(name, create_fn)] のリスト

    Returns:
        dict: {clf_name: {'mean_accuracy': float, 'std_accuracy': float}}
    """
    rs = reservoir_config['random_state']
    subjects = np.array([m['person'] for m in metadata])
    sessions = np.array([m['sample_idx'] for m in metadata])
    unique_subjects = np.unique(subjects)

    subject_accs = {name: [] for name, _ in classifiers}

    for test_subject in unique_subjects:
        subject_mask = subjects == test_subject
        subject_indices = np.where(subject_mask)[0]
        subject_sessions = sessions[subject_mask]
        unique_sessions = np.unique(subject_sessions)
        n_sessions = len(unique_sessions)

        # 元コードと同一のsubj_id計算
        try:
            subj_id = int(test_subject)
        except Exception:
            subj_id = sum(ord(c) for c in str(test_subject)) % 10000

        np.random.seed(42 + subj_id)
        shuffled_sessions = np.random.permutation(unique_sessions)
        split_point = n_sessions // 2
        train_sessions = shuffled_sessions[:split_point]
        test_sessions_set = shuffled_sessions[split_point:]

        train_mask = np.isin(subject_sessions, train_sessions)
        test_mask = np.isin(subject_sessions, test_sessions_set)
        train_idx = subject_indices[train_mask]
        test_idx = subject_indices[test_mask]

        train_states = all_states[train_idx]
        test_states = all_states[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, create_fn in classifiers:
            clf = create_fn(rs)
            clf.fit(train_states, y_train)
            y_pred = clf.predict(test_states)
            subject_accs[name].append(accuracy_score(y_test, y_pred))

    results = {}
    for name, _ in classifiers:
        accs = subject_accs[name]
        results[name] = {
            'mean_accuracy': np.mean(accs),
            'std_accuracy': np.std(accs),
        }

    return results


def evaluate_loso(all_states, y, metadata, reservoir_config, classifiers):
    """
    Leave-One-Subject-Out交差検証（元コードと同一）

    Args:
        all_states: 事前計算済みリザバー状態
        y: ラベル配列
        metadata: メタデータリスト
        reservoir_config: リザバー設定
        classifiers: [(name, create_fn)] のリスト

    Returns:
        dict: {clf_name: {'mean_accuracy': float, 'std_accuracy': float}}
    """
    rs = reservoir_config['random_state']
    subjects = np.array([m['person'] for m in metadata])
    unique_subjects = np.unique(subjects)

    subject_accs = {name: [] for name, _ in classifiers}

    for test_subject in unique_subjects:
        test_mask = subjects == test_subject
        train_mask = ~test_mask
        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        train_states = all_states[train_idx]
        test_states = all_states[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name, create_fn in classifiers:
            clf = create_fn(rs)
            clf.fit(train_states, y_train)
            y_pred = clf.predict(test_states)
            subject_accs[name].append(accuracy_score(y_test, y_pred))

    results = {}
    for name, _ in classifiers:
        accs = subject_accs[name]
        results[name] = {
            'mean_accuracy': np.mean(accs),
            'std_accuracy': np.std(accs),
        }

    return results


def run_dopnet_evaluation(X, y, metadata, reservoir_config, classifiers):
    """
    全評価を実行

    Args:
        X: 入力データリスト（可変長信号）
        y: ラベル配列
        metadata: メタデータリスト
        reservoir_config: リザバー設定
        classifiers: [(name, create_fn)] のリスト

    Returns:
        dict: 全評価結果
    """
    results = {}

    # 50:50, LOSO, session用にリザバー状態を一括計算（同一接続行列）
    total_start = time.time()

    # 50:50, LOSO, session用にリザバー状態を一括計算（同一接続行列）
    print(f"  リザバー状態計算中 ({len(X)}サンプル)...", end=' ', flush=True)
    all_states = _compute_all_states(X, reservoir_config)
    print("✓")

    clf_names = [name for name, _ in classifiers]

    print("  50:50分割評価中...", end=' ', flush=True)
    t0 = time.time()
    results['50_50'] = evaluate_50_50(all_states, y, reservoir_config, classifiers)
    acc_strs = [f"{name}={results['50_50'][name]*100:.2f}%" for name in clf_names]
    print(f"✓ {', '.join(acc_strs)} ({time.time()-t0:.1f}秒)")

    print("  10分割交差検証中 (各fold毎にリザバー再計算)...")
    t0 = time.time()
    results['10fold'] = evaluate_10fold(X, y, reservoir_config, classifiers)
    acc_strs = [f"{name}={results['10fold'][name]['mean_accuracy']*100:.2f}%±{results['10fold'][name]['std_accuracy']*100:.2f}%" for name in clf_names]
    print(f"  10分割交差検証 ✓ {', '.join(acc_strs)} ({time.time()-t0:.1f}秒)")

    print("  セッション分割評価中...", end=' ', flush=True)
    t0 = time.time()
    results['session_split'] = evaluate_session_split(
        all_states, y, metadata, reservoir_config, classifiers)
    acc_strs = [f"{name}={results['session_split'][name]['mean_accuracy']*100:.2f}%±{results['session_split'][name]['std_accuracy']*100:.2f}%" for name in clf_names]
    print(f"✓ {', '.join(acc_strs)} ({time.time()-t0:.1f}秒)")

    print("  LOSO評価中...", end=' ', flush=True)
    t0 = time.time()
    results['loso'] = evaluate_loso(
        all_states, y, metadata, reservoir_config, classifiers)
    acc_strs = [f"{name}={results['loso'][name]['mean_accuracy']*100:.2f}%±{results['loso'][name]['std_accuracy']*100:.2f}%" for name in clf_names]
    print(f"✓ {', '.join(acc_strs)} ({time.time()-t0:.1f}秒)")

    print(f"\n  全評価完了: {time.time()-total_start:.1f}秒")

    return results


# Soli retains its own model fitting and split seeds.
def evaluate_10fold_cv(model_class, X_md, X_rtm, y, model_params, n_splits=10,
                       method_name='', save_confusion_matrix=False):
    """
    10分割交差検証

    Args:
        model_class: モデルクラス (FeatESNReadout or ClassifierESNReadout)
        X_md: MDデータ
        X_rtm: RTMデータ
        y: ラベル
        model_params: モデル初期化パラメータ
        n_splits: 分割数
        method_name: 手法名（表示用）
        save_confusion_matrix: 混同行列を保存するか

    Returns:
        dict: 結果辞書
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_train_times = []
    fold_eval_times = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(y, y)):
        X_md_train = {ch: [X_md[ch][i] for i in train_idx] for ch in range(4)}
        X_md_val = {ch: [X_md[ch][i] for i in val_idx] for ch in range(4)}
        X_rtm_train = {ch: [X_rtm[ch][i] for i in train_idx] for ch in range(4)}
        X_rtm_val = {ch: [X_rtm[ch][i] for i in val_idx] for ch in range(4)}
        y_train = y[train_idx]
        y_val = y[val_idx]

        model = model_class(**model_params)

        verbose = False
        feature_time, train_time = model.fit(X_md_train, X_rtm_train, y_train, verbose=verbose)
        total_train_time = feature_time + train_time

        predictions, feature_time, predict_time = model.predict(X_md_val, X_rtm_val, verbose=False)
        eval_time = (feature_time + predict_time) / len(y_val)

        accuracy = accuracy_score(y_val, predictions)

        fold_accuracies.append(accuracy)
        fold_train_times.append(total_train_time)
        fold_eval_times.append(eval_time)

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    return {'mean_accuracy': mean_acc, 'std_accuracy': std_acc}


def evaluate_50_50_split(model_class, X_md, X_rtm, y, model_params,
                         method_name='', random_state=42):
    """
    50:50データ分割評価（両方向）
    """
    n_samples = len(y)
    indices = np.arange(n_samples)
    train_idx, test_idx = train_test_split(indices, test_size=0.5,
                                           stratify=y, random_state=random_state)

    # パターン1: 前半訓練 → 後半テスト
    X_md_train = {ch: [X_md[ch][i] for i in train_idx] for ch in range(4)}
    X_md_test = {ch: [X_md[ch][i] for i in test_idx] for ch in range(4)}
    X_rtm_train = {ch: [X_rtm[ch][i] for i in train_idx] for ch in range(4)}
    X_rtm_test = {ch: [X_rtm[ch][i] for i in test_idx] for ch in range(4)}
    y_train, y_test = y[train_idx], y[test_idx]

    model = model_class(**model_params)
    model.fit(X_md_train, X_rtm_train, y_train, verbose=False)
    predictions, _, _ = model.predict(X_md_test, X_rtm_test, verbose=False)
    accuracy1 = accuracy_score(y_test, predictions)

    # パターン2: 後半訓練 → 前半テスト
    X_md_train = {ch: [X_md[ch][i] for i in test_idx] for ch in range(4)}
    X_md_test = {ch: [X_md[ch][i] for i in train_idx] for ch in range(4)}
    X_rtm_train = {ch: [X_rtm[ch][i] for i in test_idx] for ch in range(4)}
    X_rtm_test = {ch: [X_rtm[ch][i] for i in train_idx] for ch in range(4)}
    y_train, y_test = y[test_idx], y[train_idx]

    model = model_class(**model_params)
    model.fit(X_md_train, X_rtm_train, y_train, verbose=False)
    predictions, _, _ = model.predict(X_md_test, X_rtm_test, verbose=False)
    accuracy2 = accuracy_score(y_test, predictions)

    return {'accuracy_pattern1': accuracy1, 'accuracy_pattern2': accuracy2}


def evaluate_leave_one_session_out(model_class, X_md, X_rtm, y, metadata,
                                   model_params, method_name=''):
    """
    被験者内セッション50:50分割（両方向）
    """
    subjects = np.array([m['subject'] for m in metadata])
    sessions = np.array([m['session'] for m in metadata])
    unique_subjects = np.unique(subjects)

    subject_accuracies = []

    # パターン1: 前半セッション訓練 → 後半セッションテスト
    for test_subject in unique_subjects:
        subject_mask = subjects == test_subject
        subject_indices = np.where(subject_mask)[0]
        subject_sessions = sessions[subject_mask]

        unique_subject_sessions = np.unique(subject_sessions)
        n_sessions = len(unique_subject_sessions)

        np.random.seed(42 + test_subject)
        shuffled_sessions = np.random.permutation(unique_subject_sessions)

        split_point = n_sessions // 2
        train_sessions = shuffled_sessions[:split_point]
        test_sessions = shuffled_sessions[split_point:]

        train_mask = np.isin(sessions[subject_mask], train_sessions)
        test_mask = np.isin(sessions[subject_mask], test_sessions)

        train_idx = subject_indices[train_mask]
        test_idx = subject_indices[test_mask]

        X_md_train = {ch: [X_md[ch][i] for i in train_idx] for ch in range(4)}
        X_md_test = {ch: [X_md[ch][i] for i in test_idx] for ch in range(4)}
        X_rtm_train = {ch: [X_rtm[ch][i] for i in train_idx] for ch in range(4)}
        X_rtm_test = {ch: [X_rtm[ch][i] for i in test_idx] for ch in range(4)}
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_class(**model_params)
        model.fit(X_md_train, X_rtm_train, y_train, verbose=False)
        predictions, _, _ = model.predict(X_md_test, X_rtm_test, verbose=False)
        accuracy = accuracy_score(y_test, predictions)
        subject_accuracies.append(accuracy)

    mean_acc = np.mean(subject_accuracies)
    std_acc = np.std(subject_accuracies)

    return {'mean_accuracy': mean_acc, 'std_accuracy': std_acc}


def evaluate_leave_one_subject_out(model_class, X_md, X_rtm, y, metadata,
                                   model_params, method_name=''):
    """
    Leave-One-Subject-Out交差検証
    """
    subjects = np.array([m['subject'] for m in metadata])
    unique_subjects = np.unique(subjects)

    fold_accuracies = []

    for test_subject in unique_subjects:
        test_mask = subjects == test_subject
        train_mask = ~test_mask

        train_idx = np.where(train_mask)[0]
        test_idx = np.where(test_mask)[0]

        X_md_train = {ch: [X_md[ch][i] for i in train_idx] for ch in range(4)}
        X_md_test = {ch: [X_md[ch][i] for i in test_idx] for ch in range(4)}
        X_rtm_train = {ch: [X_rtm[ch][i] for i in train_idx] for ch in range(4)}
        X_rtm_test = {ch: [X_rtm[ch][i] for i in test_idx] for ch in range(4)}
        y_train, y_test = y[train_idx], y[test_idx]

        model = model_class(**model_params)
        model.fit(X_md_train, X_rtm_train, y_train, verbose=False)
        predictions, _, _ = model.predict(X_md_test, X_rtm_test, verbose=False)
        accuracy = accuracy_score(y_test, predictions)
        fold_accuracies.append(accuracy)

    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)

    return {'mean_accuracy': mean_acc, 'std_accuracy': std_acc}


def run_soli_evaluation(model_class, X_md, X_rtm, y, metadata, model_params, method_name):
    """
    全評価を実行

    Args:
        model_class: モデルクラス
        X_md: MDデータ
        X_rtm: RTMデータ
        y: ラベル
        metadata: メタデータ
        model_params: モデルパラメータ
        method_name: 手法名

    Returns:
        dict: 全結果
    """
    results = {}

    print(f"  50:50分割評価中...", end=' ', flush=True)
    results['50_50'] = evaluate_50_50_split(
        model_class, X_md, X_rtm, y, model_params, method_name)
    print(f"✓ パターン1={results['50_50']['accuracy_pattern1']*100:.2f}%, パターン2={results['50_50']['accuracy_pattern2']*100:.2f}%")

    print(f"  10分割交差検証中...", end=' ', flush=True)
    results['10fold'] = evaluate_10fold_cv(
        model_class, X_md, X_rtm, y, model_params, method_name=method_name)
    print(f"✓ {results['10fold']['mean_accuracy']*100:.2f}% ± {results['10fold']['std_accuracy']*100:.2f}%")

    print(f"  セッション分割評価中...", end=' ', flush=True)
    results['session_split'] = evaluate_leave_one_session_out(
        model_class, X_md, X_rtm, y, metadata, model_params, method_name)
    print(f"✓ {results['session_split']['mean_accuracy']*100:.2f}% ± {results['session_split']['std_accuracy']*100:.2f}%")

    print(f"  LOSO評価中...", end=' ', flush=True)
    results['loso'] = evaluate_leave_one_subject_out(
        model_class, X_md, X_rtm, y, metadata, model_params, method_name)
    print(f"✓ {results['loso']['mean_accuracy']*100:.2f}% ± {results['loso']['std_accuracy']*100:.2f}%")

    return results
