#!/usr/bin/env python3
"""
ESN Gesture Classification モジュールパッケージ

モジュール構成:
- config: 共通設定（マルチリザバー・シングルリザバー）
- reservoir: ESNリザバーコンポーネント
- dataloader: データ読み込み
- multi_feat_esn_readout: マルチリザバー Ridge回帰リードアウト
- multi_classifier_readout: マルチリザバー 機械学習分類器リードアウト
- single_reservoir: シングルリザバーESN
- evaluation: 共通評価関数
- multi_RR_L, multi_RR_N, multi_SVM, multi_RF: マルチリザバー各手法
- single_RF, single_SVM, single_Ridge: シングルリザバー各手法
"""

from .config import *
from .reservoir import VariableLengthESN
from .dataloader import DualDataTypeLoader
from .multi_feat_esn_readout import FeatESNReadout
from .multi_classifier_readout import ClassifierESNReadout
from .single_reservoir import SingleReservoirESN
from .evaluation import (
    evaluate_10fold_cv,
    evaluate_50_50_split,
    evaluate_leave_one_session_out,
    evaluate_leave_one_subject_out,
    run_full_evaluation
)

# 各リードアウト手法
from . import multi_RR_L
from . import multi_RR_N
from . import multi_SVM
from . import multi_RF
from . import single_RF
from . import single_SVM
from . import single_Ridge

__all__ = [
    # 設定
    'DATA_CONFIG', 'MULTI_RESERVOIR_CONFIG', 'SINGLE_RESERVOIR_CONFIG',
    'RESERVOIR_CONFIG', 'RIDGE_READOUT_CONFIG',
    'MULTI_RR_L_CONFIG', 'MULTI_RR_N_CONFIG', 'RF_CONFIG', 'SVM_CONFIG', 'RIDGE_CONFIG',
    'RR_L_CONFIG', 'RR_N_CONFIG', 'EVAL_CONFIG',
    # クラス
    'VariableLengthESN', 'DualDataTypeLoader', 
    'FeatESNReadout', 'ClassifierESNReadout', 'SingleReservoirESN',
    # 評価関数
    'evaluate_10fold_cv', 'evaluate_50_50_split', 
    'evaluate_leave_one_session_out', 'evaluate_leave_one_subject_out',
    'run_full_evaluation',
    # リードアウト手法
    'multi_RR_L', 'multi_RR_N', 'multi_SVM', 'multi_RF',
    'single_RF', 'single_SVM', 'single_Ridge',
]
