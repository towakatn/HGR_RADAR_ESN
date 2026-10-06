#!/usr/bin/env python3
"""
RF: Classifier-Based ESN Readout (Random Forest)
Random Forest Classifier

設定はconfig.pyからインポート
評価関数はevaluation.pyを使用
"""

from .dataloader import DualDataTypeLoader
from .multi_classifier_readout import ClassifierESNReadout
from .config import DATA_CONFIG, MULTI_RESERVOIR_CONFIG, RF_CONFIG
from .evaluation import run_full_evaluation


def get_model_params():
    """Multi RF用モデルパラメータを取得"""
    return {
        'classifier_type': 'rf',
        'n_reservoir': MULTI_RESERVOIR_CONFIG['n_reservoir'],
        'spectral_radius': MULTI_RESERVOIR_CONFIG['spectral_radius'],
        'input_scaling': MULTI_RESERVOIR_CONFIG['input_scaling'],
        'density': MULTI_RESERVOIR_CONFIG['density'],
        'leakage_rate': MULTI_RESERVOIR_CONFIG['leakage_rate'],
        'bias_scaling': MULTI_RESERVOIR_CONFIG['bias_scaling'],
        'random_state': MULTI_RESERVOIR_CONFIG['random_state'],
        'classifier_config': RF_CONFIG,
    }


def run_evaluation(X_md, X_rtm, y, metadata):
    """
    RF評価を実行
    
    Args:
        X_md: MDデータ
        X_rtm: RTMデータ
        y: ラベル
        metadata: メタデータ
    
    Returns:
        dict: 評価結果
    """
    model_params = get_model_params()
    method_name = RF_CONFIG['name']
    
    return run_full_evaluation(
        ClassifierESNReadout, X_md, X_rtm, y, metadata, model_params, method_name)


def main():
    """スタンドアロン実行用"""
    loader = DualDataTypeLoader(
        channels=DATA_CONFIG['channels'],
        base_dir=DATA_CONFIG['base_dir']
    )
    X_md, X_rtm, y, metadata = loader.load_gesture_data(
        max_samples_per_gesture_subject=DATA_CONFIG['max_samples_per_gesture_subject']
    )
    
    run_evaluation(X_md, X_rtm, y, metadata)


if __name__ == '__main__':
    main()
