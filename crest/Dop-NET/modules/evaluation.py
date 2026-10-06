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

import sys
import os
import time
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score

# 親ディレクトリからインポート
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.reservoir_computer import ReservoirComputer


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


def run_full_evaluation(X, y, metadata, reservoir_config, classifiers):
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
