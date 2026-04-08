#!/usr/bin/env python3
"""
RR_N: Feature-Based ESN Readout (Nonlinear)
Ψ(r) = [1, r^T, tanh(r)^T]^T

設定はconfig.pyからインポート
評価関数はevaluation.pyを使用
"""

from .dataloader import DualDataTypeLoader
from .multi_feat_esn_readout import FeatESNReadout
from .config import DATA_CONFIG, MULTI_RESERVOIR_CONFIG, RIDGE_READOUT_CONFIG, MULTI_RR_N_CONFIG
from .evaluation import run_full_evaluation


def get_model_params():
    """Multi RR_N用モデルパラメータを取得"""
    return {
        'n_reservoir_per_stream': MULTI_RESERVOIR_CONFIG['n_reservoir'],
        'n_selected_nodes': MULTI_RESERVOIR_CONFIG['n_reservoir'],
        'spectral_radius': MULTI_RESERVOIR_CONFIG['spectral_radius'],
        'input_scaling': MULTI_RESERVOIR_CONFIG['input_scaling'],
        'density': MULTI_RESERVOIR_CONFIG['density'],
        'leakage_rate': MULTI_RESERVOIR_CONFIG['leakage_rate'],
        'bias_scaling': MULTI_RESERVOIR_CONFIG['bias_scaling'],
        'regularization': RIDGE_READOUT_CONFIG['regularization'],
        'nonlinear_features': MULTI_RR_N_CONFIG['nonlinear_features'],
        'random_state': MULTI_RESERVOIR_CONFIG['random_state'],
    }


def run_evaluation(X_md, X_rtm, y, metadata):
    """
    RR_N評価を実行
    
    Args:
        X_md: MDデータ
        X_rtm: RTMデータ
        y: ラベル
        metadata: メタデータ
    
    Returns:
        dict: 評価結果
    """
    model_params = get_model_params()
    method_name = MULTI_RR_N_CONFIG['name']
    
    return run_full_evaluation(
        FeatESNReadout, X_md, X_rtm, y, metadata, model_params, method_name)


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
