#!/usr/bin/env python3
"""
全リードアウト手法の一括評価スクリプト

シングルリザバー（1000ノード）× 3リードアウト:
1. RF: Random Forest (n_estimators=300)
2. SVM: Support Vector Machine (RBF kernel, C=10.0)
3. Ridge: Ridge Classifier (alpha=1.0)

リザバー状態を共有して効率的に全手法を評価
"""

import sys
import os
import time
from datetime import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules.data_loader import RCDataLoader
from modules.reservoir_computer import prepare_rc_input
from modules import RF, SVM, Ridge
from modules.config import RESERVOIR_CONFIG, DATA_CONFIG
from modules.evaluation import run_full_evaluation


def main():
    print("=" * 80)
    print("全リードアウト手法の包括的評価")
    print("=" * 80)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # データ読み込み
    print("\n  データ読み込み中...", end=' ', flush=True)
    loader = RCDataLoader(data_dir=DATA_CONFIG['data_dir'])
    signals, labels, metadata = loader.load_all_data()
    X = prepare_rc_input(signals)
    y = np.array(labels)
    print(f"✓ ({len(X)}サンプル, {len(np.unique(y))}クラス)")

    # 分類器リストを定義
    classifiers = [
        (RF.get_name(), RF.create_classifier),
        (SVM.get_name(), SVM.create_classifier),
        (Ridge.get_name(), Ridge.create_classifier),
    ]

    # 全評価実行
    print()
    start_time = time.time()
    results = run_full_evaluation(X, y, metadata, RESERVOIR_CONFIG, classifiers)
    total_time = time.time() - start_time

    # サマリー表示
    clf_names = [name for name, _ in classifiers]

    print("\n" + "=" * 80)
    print("全手法の結果サマリー")
    print("=" * 80)
    print()

    # ヘッダー
    print(f"{'手法':<10} {'50:50':<10} {'10-Fold CV':<15} {'Session':<15} {'LOSO':<15}")
    print("-" * 65)

    for name in clf_names:
        acc_50 = results['50_50'][name] * 100
        cv_mean = results['10fold'][name]['mean_accuracy'] * 100
        cv_std = results['10fold'][name]['std_accuracy'] * 100
        sess_mean = results['session_split'][name]['mean_accuracy'] * 100
        sess_std = results['session_split'][name]['std_accuracy'] * 100
        loso_mean = results['loso'][name]['mean_accuracy'] * 100
        loso_std = results['loso'][name]['std_accuracy'] * 100

        print(f"{name:<10} {acc_50:5.2f}%    {cv_mean:5.2f}±{cv_std:4.2f}%  {sess_mean:5.2f}±{sess_std:4.2f}%  {loso_mean:5.2f}±{loso_std:4.2f}%")

    print("-" * 65)

    # 各評価での最高精度
    print()
    best_50 = max(clf_names, key=lambda n: results['50_50'][n])
    best_cv = max(clf_names, key=lambda n: results['10fold'][n]['mean_accuracy'])
    best_sess = max(clf_names, key=lambda n: results['session_split'][n]['mean_accuracy'])
    best_loso = max(clf_names, key=lambda n: results['loso'][n]['mean_accuracy'])

    print(f"最高精度:")
    print(f"  50:50:   {best_50} ({results['50_50'][best_50]*100:.2f}%)")
    print(f"  10-Fold: {best_cv} ({results['10fold'][best_cv]['mean_accuracy']*100:.2f}%)")
    print(f"  Session: {best_sess} ({results['session_split'][best_sess]['mean_accuracy']*100:.2f}%)")
    print(f"  LOSO:    {best_loso} ({results['loso'][best_loso]['mean_accuracy']*100:.2f}%)")

    print()
    print(f"実行時間: {total_time:.2f}秒")
    print("=" * 80)
    print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    return results


if __name__ == '__main__':
    main()
