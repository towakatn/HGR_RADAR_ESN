#!/usr/bin/env python3
"""
Single RF: Single-Reservoir ESN with Random Forest
1つのリザバー（500ノード）でRandom Forest分類

設定はconfig.pyからインポート
評価関数はevaluation.pyを使用
"""

from .dataloader import DualDataTypeLoader
from .single_reservoir import SingleReservoirESN
from .config import DATA_CONFIG, SINGLE_RESERVOIR_CONFIG, RF_CONFIG
from .evaluation import run_full_evaluation


def get_model_params():
    """Single RF用モデルパラメータを取得"""
    return {
        'classifier_type': 'rf',
        'n_reservoir': SINGLE_RESERVOIR_CONFIG['n_reservoir'],
        'spectral_radius': SINGLE_RESERVOIR_CONFIG['spectral_radius'],
        'input_scaling': SINGLE_RESERVOIR_CONFIG['input_scaling'],
        'density': SINGLE_RESERVOIR_CONFIG['density'],
        'leakage_rate': SINGLE_RESERVOIR_CONFIG['leakage_rate'],
        'bias_scaling': SINGLE_RESERVOIR_CONFIG['bias_scaling'],
        'node_selection_ratio': SINGLE_RESERVOIR_CONFIG['node_selection_ratio'],
        'random_state': SINGLE_RESERVOIR_CONFIG['random_state'],
        'classifier_config': RF_CONFIG,
    }


def run_evaluation(X_md, X_rtm, y, metadata):
    """
    Single RF評価を実行
    
    Args:
        X_md: MDデータ
        X_rtm: RTMデータ
        y: ラベル
        metadata: メタデータ
    
    Returns:
        dict: 評価結果
    """
    model_params = get_model_params()
    method_name = "Single_RF"
    
    return run_full_evaluation(
        SingleReservoirESN, X_md, X_rtm, y, metadata, model_params, method_name)


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
