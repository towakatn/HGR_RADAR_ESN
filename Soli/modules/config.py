#!/usr/bin/env python3
"""
共通設定ファイル
マルチリザバー（8リザバー）とシングルリザバー（1リザバー）の設定を管理
"""

# ================================================================================
# データ設定
# ================================================================================

DATA_CONFIG = {
    'base_dir': '.',  # データディレクトリのベースパス（Soli/から実行）
    'channels': [0, 1, 2, 3],  # 使用するチャンネル
    'max_samples_per_gesture_subject': 25,  # 各ジェスチャー・被験者の最大サンプル数
}

# ================================================================================
# マルチリザバー設定（8リザバー: 4チャンネル × 2データタイプ）
# ================================================================================

MULTI_RESERVOIR_CONFIG = {
    'n_reservoir': 50,  # 各リザバーのノード数
    'spectral_radius': 0.95,  # スペクトル半径
    'input_scaling': 1.0,  # 入力スケーリング
    'density': 0.9,  # リザバー接続密度
    'leakage_rate': 0.0263,  # リーク率
    'bias_scaling': 0.0,  # バイアススケーリング
    'random_state': 42,  # 乱数シード
}

# ================================================================================
# シングルリザバー設定（1リザバー: 全データを結合）
# ================================================================================

SINGLE_RESERVOIR_CONFIG = {
    'n_reservoir': 500,  # リザバーノード数
    'spectral_radius': 0.95,  # スペクトル半径
    'input_scaling': 0.2,  # 入力スケーリング
    'density': 0.1,  # リザバー接続密度
    'leakage_rate': 0.05,  # リーク率
    'bias_scaling': 0.05,  # バイアススケーリング
    'node_selection_ratio': 1.0,  # ノード選択率（100%）
    'random_state': 42,  # 乱数シード
}

# 下位互換性のため（既存コード用）
RESERVOIR_CONFIG = MULTI_RESERVOIR_CONFIG

# ================================================================================
# ================================================================================
# マルチリザバー用Ridge回帰リードアウト設定 (multi_RR_L, multi_RR_N)
# ================================================================================

RIDGE_READOUT_CONFIG = {
    'regularization': 0.001,  # Tikhonov正則化係数 (λ)
}

# Multi RR_L (Linear): Ψ(r) = r
MULTI_RR_L_CONFIG = {
    'nonlinear_features': 'none',
    'name': 'Multi_RR_L',
    'description': 'Multi-Reservoir Ridge Regression (Linear) - Ψ(r) = r',
}

# Multi RR_N (Nonlinear): Ψ(r) = [1, r^T, tanh(r)^T]^T
MULTI_RR_N_CONFIG = {
    'nonlinear_features': 'square_tanh',
    'name': 'Multi_RR_N',
    'description': 'Multi-Reservoir Ridge Regression (Nonlinear) - Ψ(r) = [1, r, tanh(r)]',
}

# 下位互換性のため
RR_L_CONFIG = MULTI_RR_L_CONFIG
RR_N_CONFIG = MULTI_RR_N_CONFIG

# ================================================================================
# 機械学習分類器設定 (SVM, RF, Ridge)
# ================================================================================

# Random Forest設定
RF_CONFIG = {
    'n_estimators': 300,
    'max_depth': None,
    'random_state': 42,
    'n_jobs': -1,
    'name': 'Multi_RF',
    'description': 'Multi-Reservoir Random Forest Classifier',
}

# SVM (RBF kernel)設定
SVM_CONFIG = {
    'kernel': 'rbf',
    'C': 10.0,
    'gamma': 'scale',
    'random_state': 42,
    'name': 'Multi_SVM',
    'description': 'Multi-Reservoir Support Vector Machine (RBF kernel)',
}

# Ridge Classifier設定（シングルリザバー用）
RIDGE_CONFIG = {
    'alpha': 1.0,
    'random_state': 42,
    'name': 'Ridge',
    'description': 'Ridge Classifier',
}

# ================================================================================
# 評価設定
# ================================================================================

EVAL_CONFIG = {
    'n_splits': 10,  # K分割交差検証の分割数
    'test_size': 0.5,  # 50:50分割のテスト比率
}

# ================================================================================
# ヘルパー関数
# ================================================================================

def get_reservoir_params():
    """リザバーパラメータを取得"""
    return RESERVOIR_CONFIG.copy()

def get_ridge_params():
    """Ridge回帰パラメータを取得"""
    params = RESERVOIR_CONFIG.copy()
    params.update(RIDGE_READOUT_CONFIG)
    return params

def get_rr_l_params():
    """RR_L用の全パラメータを取得"""
    params = get_ridge_params()
    params.update(RR_L_CONFIG)
    return params

def get_rr_n_params():
    """RR_N用の全パラメータを取得"""
    params = get_ridge_params()
    params.update(RR_N_CONFIG)
    return params

def get_rf_params():
    """Random Forest用パラメータを取得"""
    params = {
        'reservoir': RESERVOIR_CONFIG.copy(),
        'classifier': RF_CONFIG.copy(),
    }
    return params

def get_svm_params():
    """SVM用パラメータを取得"""
    params = {
        'reservoir': RESERVOIR_CONFIG.copy(),
        'classifier': SVM_CONFIG.copy(),
    }
    return params
