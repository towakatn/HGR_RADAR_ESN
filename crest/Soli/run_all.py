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
import numpy as np

from modules.dataloader import DualDataTypeLoader
from modules.config import DATA_CONFIG, MULTI_RESERVOIR_CONFIG, SINGLE_RESERVOIR_CONFIG

# 各手法のモジュールをインポート
from modules import multi_RR_L
from modules import multi_RR_N
from modules import multi_SVM
from modules import multi_RF
from modules import single_RF
from modules import single_SVM
from modules import single_Ridge


def main():
    print("=" * 80)
    print("全リードアウト手法の包括的評価")
    print("=" * 80)
    print(f"開始時刻: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # データ読み込み（1回だけ）
    loader = DualDataTypeLoader(
        channels=DATA_CONFIG['channels'],
        base_dir=DATA_CONFIG['base_dir']
    )
    X_md, X_rtm, y, metadata = loader.load_gesture_data(
        max_samples_per_gesture_subject=DATA_CONFIG['max_samples_per_gesture_subject']
    )
    print()
    
    # 結果格納
    all_results = {}
    
    # マルチリザバー評価
    print("\n【1/7】Multi_RR_L")
    all_results['Multi_RR_L'] = multi_RR_L.run_evaluation(X_md, X_rtm, y, metadata)
    
    print("\n【2/7】Multi_RR_N")
    all_results['Multi_RR_N'] = multi_RR_N.run_evaluation(X_md, X_rtm, y, metadata)
    
    print("\n【3/7】Multi_SVM")
    all_results['Multi_SVM'] = multi_SVM.run_evaluation(X_md, X_rtm, y, metadata)
    
    print("\n【4/7】Multi_RF")
    all_results['Multi_RF'] = multi_RF.run_evaluation(X_md, X_rtm, y, metadata)
    
    # シングルリザバー評価
    print("\n【5/7】Single_RF")
    all_results['Single_RF'] = single_RF.run_evaluation(X_md, X_rtm, y, metadata)
    
    print("\n【6/7】Single_SVM")
    all_results['Single_SVM'] = single_SVM.run_evaluation(X_md, X_rtm, y, metadata)
    
    print("\n【7/7】Single_Ridge")
    all_results['Single_Ridge'] = single_Ridge.run_evaluation(X_md, X_rtm, y, metadata)
    
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
    
    return all_results


if __name__ == '__main__':
    main()
