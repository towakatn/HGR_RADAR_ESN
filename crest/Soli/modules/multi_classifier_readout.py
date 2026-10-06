#!/usr/bin/env python3
"""
Classifier-Based ESN Readout
SVM, Random Forestなどの機械学習分類器を使用したリードアウト
reservoir.pyのVariableLengthESNを使用
"""

import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from .reservoir import VariableLengthESN
from .config import RESERVOIR_CONFIG, RF_CONFIG, SVM_CONFIG


class ClassifierESNReadout:
    """
    Classifier-Based ESN Readout
    
    マルチリザバー構造:
    - MDData（Doppler-Time）4チャンネル × RTMData（Range-Time）4チャンネル = 8リザバー
    - 各リザバーの最終状態を統合し、分類器に入力
    
    対応分類器:
    - 'rf': Random Forest (300 estimators)
    - 'svm': SVM (RBF kernel, C=10.0)
    """
    
    def __init__(self, classifier_type='rf',
                 n_reservoir=None, spectral_radius=None, input_scaling=None,
                 density=None, leakage_rate=None, bias_scaling=None,
                 random_state=None,
                 classifier_config=None):
        """
        Classifier-Based ESN Readout の初期化
        
        Args:
            classifier_type: 'rf' or 'svm'
            n_reservoir: リザバーノード数（Noneの場合はconfig.pyから取得）
            spectral_radius: スペクトル半径
            input_scaling: 入力スケーリング
            density: リザバー接続密度
            leakage_rate: リーク率
            bias_scaling: バイアススケーリング
            random_state: 乱数シード
            classifier_config: 分類器固有の設定（Noneの場合はconfig.pyから取得）
        """
        # config.pyからデフォルト値を取得
        self.n_reservoir = n_reservoir if n_reservoir is not None else RESERVOIR_CONFIG['n_reservoir']
        self.spectral_radius = spectral_radius if spectral_radius is not None else RESERVOIR_CONFIG['spectral_radius']
        self.input_scaling = input_scaling if input_scaling is not None else RESERVOIR_CONFIG['input_scaling']
        self.density = density if density is not None else RESERVOIR_CONFIG['density']
        self.leakage_rate = leakage_rate if leakage_rate is not None else RESERVOIR_CONFIG['leakage_rate']
        self.bias_scaling = bias_scaling if bias_scaling is not None else RESERVOIR_CONFIG['bias_scaling']
        self.random_state = random_state if random_state is not None else RESERVOIR_CONFIG['random_state']
        
        self.classifier_type = classifier_type
        
        # 8リザバーの作成（4チャンネル × 2データタイプ）
        # 重要: random_stateの設定は feat_esn_readout.py と同じ
        # MD: random_state + ch
        # RTM: random_state + ch + 100
        self.esns_md = {}
        self.esns_rtm = {}
        
        for ch in range(4):
            self.esns_md[ch] = VariableLengthESN(
                n_reservoir=self.n_reservoir,
                spectral_radius=self.spectral_radius,
                input_scaling=self.input_scaling,
                density=self.density,
                leakage_rate=self.leakage_rate,
                bias_scaling=self.bias_scaling,
                random_state=self.random_state + ch
            )
            
            self.esns_rtm[ch] = VariableLengthESN(
                n_reservoir=self.n_reservoir,
                spectral_radius=self.spectral_radius,
                input_scaling=self.input_scaling,
                density=self.density,
                leakage_rate=self.leakage_rate,
                bias_scaling=self.bias_scaling,
                random_state=self.random_state + ch + 100
            )
        
        # 分類器の作成
        if classifier_type == 'rf':
            cfg = classifier_config if classifier_config else RF_CONFIG
            self.classifier = RandomForestClassifier(
                n_estimators=cfg.get('n_estimators', 300),
                max_depth=cfg.get('max_depth', None),
                random_state=cfg.get('random_state', 42),
                n_jobs=cfg.get('n_jobs', -1)
            )
        elif classifier_type == 'svm':
            cfg = classifier_config if classifier_config else SVM_CONFIG
            self.classifier = SVC(
                kernel=cfg.get('kernel', 'rbf'),
                C=cfg.get('C', 10.0),
                gamma=cfg.get('gamma', 'scale'),
                random_state=cfg.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def _extract_reservoir_states(self, X_md_channels, X_rtm_channels, verbose=False):
        """
        全チャンネル・全データタイプからリザバー状態を抽出して統合
        
        Returns:
            features: 統合リザバー状態 [n_samples, n_reservoir * 8]
        """
        all_states = []
        
        # MD（Doppler-Time）データの処理
        for ch in range(4):
            states = self.esns_md[ch].transform_sequences(X_md_channels[ch])
            all_states.append(states)
            if verbose:
                print(f"  MD Ch{ch}: {states.shape}")
        
        # RTM（Range-Time）データの処理
        for ch in range(4):
            states = self.esns_rtm[ch].transform_sequences(X_rtm_channels[ch])
            all_states.append(states)
            if verbose:
                print(f"  RTM Ch{ch}: {states.shape}")
        
        features = np.hstack(all_states)
        
        if verbose:
            print(f"  統合リザバー状態: {features.shape}")
        
        return features
    
    def extract_features(self, X_md_channels, X_rtm_channels, verbose=False):
        """特徴抽出のみを行う（fit_from_features用）"""
        return self._extract_reservoir_states(X_md_channels, X_rtm_channels, verbose=verbose)
    
    def fit(self, X_md_channels, X_rtm_channels, y, verbose=False):
        """
        分類器の学習
        
        Returns:
            (feature_time, train_time): 各処理の時間
        """
        start_time = time.time()
        features = self._extract_reservoir_states(X_md_channels, X_rtm_channels, verbose=verbose)
        feature_time = time.time() - start_time
        
        start_time = time.time()
        self.classifier.fit(features, y)
        train_time = time.time() - start_time
        
        if verbose:
            print(f"特徴抽出時間: {feature_time:.4f}秒")
            print(f"分類器学習時間: {train_time:.4f}秒")
        
        return feature_time, train_time
    
    def fit_from_features(self, features, y, return_breakdown=False):
        """事前抽出された特徴から訓練（高速化用）"""
        start_time = time.time()
        self.classifier.fit(features, y)
        train_time = time.time() - start_time
        
        if return_breakdown:
            return train_time
        return train_time
    
    def predict(self, X_md_channels, X_rtm_channels, verbose=False):
        """
        予測
        
        Returns:
            (predictions, feature_time, predict_time)
        """
        start_time = time.time()
        features = self._extract_reservoir_states(X_md_channels, X_rtm_channels, verbose=verbose)
        feature_time = time.time() - start_time
        
        start_time = time.time()
        predictions = self.classifier.predict(features)
        predict_time = time.time() - start_time
        
        return predictions, feature_time, predict_time
    
    def predict_from_features(self, features, return_breakdown=False):
        """事前抽出された特徴から予測（高速化用）"""
        start_time = time.time()
        predictions = self.classifier.predict(features)
        predict_time = time.time() - start_time
        
        if return_breakdown:
            return predictions, predict_time
        return predictions
