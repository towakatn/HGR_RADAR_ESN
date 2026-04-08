#!/usr/bin/env python3
"""
共通評価モジュール
全リードアウト手法で使用する評価関数を提供
"""

import numpy as np
from datetime import datetime
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns


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


def run_full_evaluation(model_class, X_md, X_rtm, y, metadata, model_params, method_name):
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
