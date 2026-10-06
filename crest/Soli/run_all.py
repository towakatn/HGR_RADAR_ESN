#!/usr/bin/env python3
"""
全リードアウト手法の一括評価スクリプト

マルチリザバー（8リザバー: 4チャンネル × 2データタイプ）:
1. Multi_RR_L: Ridge Regression (Linear) - Ψ(r) = r
2. Multi_RR_N: Ridge Regression (Nonlinear) - Ψ(r) = [1, r, tanh(r)]
3. Multi_SVM: Support Vector Machine (RBF kernel)
4. Multi_RF: Random Forest

シングルリザバー（1リザバー: 全データ結合）:
5. Single_RF: Random Forest
6. Single_SVM: Support Vector Machine (RBF kernel)
7. Single_Ridge: Ridge Classifier

1つのデータ読み込みで全手法を評価
"""

from datetime import datetime
from pathlib import Path
import sys
import numpy as np

DATASET_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(DATASET_DIR))
sys.path.insert(0, str(DATASET_DIR.parent))

from modules.data_loaders import DualDataTypeLoader
from modules.readouts import FeatESNReadout, ClassifierESNReadout, SingleReservoirESN
from modules.evaluation import run_soli_evaluation
from run_fusion import main as run_fusion_comparison
from soli_config import (
    DATA_CONFIG, MULTI_RESERVOIR_CONFIG, SINGLE_RESERVOIR_CONFIG,
    RF_CONFIG, SVM_CONFIG, RIDGE_CONFIG, RIDGE_READOUT_CONFIG,
    MULTI_RR_L_CONFIG, MULTI_RR_N_CONFIG,
)


def get_methods(multi_reservoir_config=None, single_reservoir_config=None):
    """Build model arguments here; shared modules have no dataset configuration imports."""
    multi = MULTI_RESERVOIR_CONFIG if multi_reservoir_config is None else multi_reservoir_config
    single = SINGLE_RESERVOIR_CONFIG if single_reservoir_config is None else single_reservoir_config
    reservoir_keys = (
        'n_reservoir', 'spectral_radius', 'input_scaling', 'density',
        'leakage_rate', 'bias_scaling', 'random_state',
    )
    multi_params = {key: multi[key] for key in reservoir_keys}
    single_params = {key: single[key] for key in (*reservoir_keys, 'node_selection_ratio')}
    ridge_params = dict(multi_params)
    ridge_params['n_reservoir_per_stream'] = ridge_params.pop('n_reservoir')
    ridge_params['n_selected_nodes'] = multi['n_reservoir']
    ridge_params['regularization'] = RIDGE_READOUT_CONFIG['regularization']
    return [
        ('Multi_RR_L', FeatESNReadout,
         dict(ridge_params, nonlinear_features=MULTI_RR_L_CONFIG['nonlinear_features']),
         MULTI_RR_L_CONFIG['name']),
        ('Multi_RR_N', FeatESNReadout,
         dict(ridge_params, nonlinear_features=MULTI_RR_N_CONFIG['nonlinear_features']),
         MULTI_RR_N_CONFIG['name']),
        ('Multi_SVM', ClassifierESNReadout,
         dict(multi_params, classifier_type='svm', classifier_config=SVM_CONFIG), SVM_CONFIG['name']),
        ('Multi_RF', ClassifierESNReadout,
         dict(multi_params, classifier_type='rf', classifier_config=RF_CONFIG), RF_CONFIG['name']),
        ('Single_RF', SingleReservoirESN,
         dict(single_params, classifier_type='rf', classifier_config=RF_CONFIG), 'Single_RF'),
        ('Single_SVM', SingleReservoirESN,
         dict(single_params, classifier_type='svm', classifier_config=SVM_CONFIG), 'Single_SVM'),
        ('Single_Ridge', SingleReservoirESN,
         dict(single_params, classifier_type='ridge', classifier_config=RIDGE_CONFIG), 'Single_Ridge'),
    ]


def main(data_config=None, multi_reservoir_config=None, single_reservoir_config=None,
         include_fusion=True, fusion_config=None):
    print("=" * 80)
    print("全リードアウト手法の包括的評価")
    print("=" * 80)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    data_config = DATA_CONFIG if data_config is None else data_config

    # データ読み込み（1回だけ）
    loader = DualDataTypeLoader(
        channels=data_config['channels'],
        base_dir=data_config['base_dir']
    )
    X_md, X_rtm, y, metadata = loader.load_gesture_data(
        max_samples_per_gesture_subject=data_config['max_samples_per_gesture_subject']
    )
    print()

    # 結果格納
    all_results = {}

    methods = get_methods(multi_reservoir_config, single_reservoir_config)
    for index, (name, model_class, model_params, method_name) in enumerate(methods, start=1):
        print(f"\n【{index}/{len(methods)}】{name}")
        all_results[name] = run_soli_evaluation(
            model_class, X_md, X_rtm, y, metadata, model_params, method_name
        )

    # 最終サマリー
    print("\n\n" + "=" * 80)
    print("全手法の結果サマリー")
    print("=" * 80)
    print()

    # ヘッダー
    print(f"{'手法':<15} {'50:50-P1':<10} {'50:50-P2':<10} {'10-Fold CV':<15} {'Session':<15} {'LOSO':<15}")
    print("-" * 80)

    # マルチリザバー
    for method_name in ['Multi_RR_L', 'Multi_RR_N', 'Multi_SVM', 'Multi_RF']:
        results = all_results[method_name]
        p1 = results['50_50']['accuracy_pattern1'] * 100
        p2 = results['50_50']['accuracy_pattern2'] * 100
        cv_mean = results['10fold']['mean_accuracy'] * 100
        cv_std = results['10fold']['std_accuracy'] * 100
        session_mean = results['session_split']['mean_accuracy'] * 100
        session_std = results['session_split']['std_accuracy'] * 100
        loso_mean = results['loso']['mean_accuracy'] * 100
        loso_std = results['loso']['std_accuracy'] * 100

        print(f"{method_name:<15} {p1:5.2f}%     {p2:5.2f}%     {cv_mean:5.2f}±{cv_std:4.2f}%  {session_mean:5.2f}±{session_std:4.2f}%  {loso_mean:5.2f}±{loso_std:4.2f}%")

    # シングルリザバー
    for method_name in ['Single_RF', 'Single_SVM', 'Single_Ridge']:
        results = all_results[method_name]
        p1 = results['50_50']['accuracy_pattern1'] * 100
        p2 = results['50_50']['accuracy_pattern2'] * 100
        cv_mean = results['10fold']['mean_accuracy'] * 100
        cv_std = results['10fold']['std_accuracy'] * 100
        session_mean = results['session_split']['mean_accuracy'] * 100
        session_std = results['session_split']['std_accuracy'] * 100
        loso_mean = results['loso']['mean_accuracy'] * 100
        loso_std = results['loso']['std_accuracy'] * 100

        print(f"{method_name:<15} {p1:5.2f}%     {p2:5.2f}%     {cv_mean:5.2f}±{cv_std:4.2f}%  {session_mean:5.2f}±{session_std:4.2f}%  {loso_mean:5.2f}±{loso_std:4.2f}%")

    print("-" * 80)

    # 各評価での最高精度
    print()
    best_50_p1 = max(all_results.items(), key=lambda x: x[1]['50_50']['accuracy_pattern1'])
    best_50_p2 = max(all_results.items(), key=lambda x: x[1]['50_50']['accuracy_pattern2'])
    best_cv = max(all_results.items(), key=lambda x: x[1]['10fold']['mean_accuracy'])
    best_session = max(all_results.items(), key=lambda x: x[1]['session_split']['mean_accuracy'])
    best_loso = max(all_results.items(), key=lambda x: x[1]['loso']['mean_accuracy'])

    print(f"最高精度:")
    print(f"  50:50-P1: {best_50_p1[0]} ({best_50_p1[1]['50_50']['accuracy_pattern1']*100:.2f}%)")
    print(f"  50:50-P2: {best_50_p2[0]} ({best_50_p2[1]['50_50']['accuracy_pattern2']*100:.2f}%)")
    print(f"  10-Fold:  {best_cv[0]} ({best_cv[1]['10fold']['mean_accuracy']*100:.2f}%)")
    print(f"  Session:  {best_session[0]} ({best_session[1]['session_split']['mean_accuracy']*100:.2f}%)")
    print(f"  LOSO:     {best_loso[0]} ({best_loso[1]['loso']['mean_accuracy']*100:.2f}%)")

    print()
    print("=" * 80)
    print(f"終了時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    if include_fusion:
        all_results['fusion_comparison'] = run_fusion_comparison(
            data_config=data_config, fusion_config=fusion_config,
            loaded_data=(X_md, X_rtm, y, metadata),
        )

    return all_results


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Run legacy Soli methods and the controlled fusion suite.')
    parser.add_argument('--legacy-only', action='store_true', help='run only the original seven methods')
    main(include_fusion=not parser.parse_args().legacy_only)
