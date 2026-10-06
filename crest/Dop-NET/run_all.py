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
from pathlib import Path
import time
from datetime import datetime
import numpy as np

DATASET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATASET_DIR))
sys.path.insert(0, str(DATASET_DIR.parent))

from modules.data_loaders import RCDataLoader
from modules.reservoir_computer import prepare_rc_input
from modules.classifiers import classifier_factory
from dopnet_config import RESERVOIR_CONFIG, DATA_CONFIG, RF_CONFIG, SVM_CONFIG, RIDGE_CONFIG
from modules.evaluation import run_dopnet_evaluation


def main(data_config=None, reservoir_config=None):
    print("=" * 80)
    print("全リードアウト手法の包括的評価")
    print("=" * 80)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    data_config = DATA_CONFIG if data_config is None else data_config
    reservoir_config = RESERVOIR_CONFIG if reservoir_config is None else reservoir_config

    # データ読み込み
    print("\n  データ読み込み中...", end=' ', flush=True)
    loader = RCDataLoader(data_dir=data_config['data_dir'])
    signals, labels, metadata = loader.load_all_data()
    X = prepare_rc_input(signals)
    y = np.array(labels)
    print(f"✓ ({len(X)}サンプル, {len(np.unique(y))}クラス)")

    # 分類器リストを定義
    classifiers = [
        (RF_CONFIG['name'], classifier_factory('rf', {
            key: RF_CONFIG[key] for key in ('n_estimators', 'n_jobs')
        })),
        (SVM_CONFIG['name'], classifier_factory('svm', {
            key: SVM_CONFIG[key] for key in ('kernel', 'C', 'gamma')
        })),
        (RIDGE_CONFIG['name'], classifier_factory(
            'ridge', {'alpha': RIDGE_CONFIG['alpha']}, use_random_state=False
        )),
    ]

    # 全評価実行
    print()
    start_time = time.time()
    results = run_dopnet_evaluation(X, y, metadata, reservoir_config, classifiers)
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
