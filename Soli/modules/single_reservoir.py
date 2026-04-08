#!/usr/bin/env python3
"""
シングルリザバーESN
全チャンネル・全データタイプを時系列で結合し、1つのリザバーで処理
"""

import numpy as np
import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import RidgeClassifier
from .reservoir import VariableLengthESN
from .config import SINGLE_RESERVOIR_CONFIG, RF_CONFIG, SVM_CONFIG, RIDGE_CONFIG


class SingleReservoirESN:
    """
    全チャンネル・全データタイプを結合して1つのリザバーで処理するESNクラス
    """
    
    def __init__(self, channels=[0, 1, 2, 3], 
                 n_reservoir=None, spectral_radius=None, input_scaling=None,
                 density=None, leakage_rate=None, bias_scaling=None,
                 node_selection_ratio=None,
                 classifier_type='rf', random_state=None,
                 classifier_config=None):
        """
        Args:
            channels: 使用するチャンネル
            n_reservoir: リザバーノード数（Noneの場合はconfig.pyから取得）
            spectral_radius: スペクトル半径
            input_scaling: 入力スケーリング
            density: 結合密度
            leakage_rate: リーク率
            bias_scaling: バイアススケーリング
            node_selection_ratio: ノード選択率
            classifier_type: 分類器タイプ ('rf', 'svm', 'ridge')
            random_state: 乱数シード
            classifier_config: 分類器固有の設定
        """
        self.channels = channels
        
        # config.pyからデフォルト値を取得
        self.n_reservoir = n_reservoir if n_reservoir is not None else SINGLE_RESERVOIR_CONFIG['n_reservoir']
        self.spectral_radius = spectral_radius if spectral_radius is not None else SINGLE_RESERVOIR_CONFIG['spectral_radius']
        self.input_scaling = input_scaling if input_scaling is not None else SINGLE_RESERVOIR_CONFIG['input_scaling']
        self.density = density if density is not None else SINGLE_RESERVOIR_CONFIG['density']
        self.leakage_rate = leakage_rate if leakage_rate is not None else SINGLE_RESERVOIR_CONFIG['leakage_rate']
        self.bias_scaling = bias_scaling if bias_scaling is not None else SINGLE_RESERVOIR_CONFIG['bias_scaling']
        self.node_selection_ratio = node_selection_ratio if node_selection_ratio is not None else SINGLE_RESERVOIR_CONFIG['node_selection_ratio']
        self.random_state = random_state if random_state is not None else SINGLE_RESERVOIR_CONFIG['random_state']
        
        self.classifier_type = classifier_type
        
        # 単一のESNを作成
        self.esn = VariableLengthESN(
            n_reservoir=self.n_reservoir,
            spectral_radius=self.spectral_radius,
            input_scaling=self.input_scaling,
            density=self.density,
            leakage_rate=self.leakage_rate,
            bias_scaling=self.bias_scaling,
            random_state=self.random_state
        )
        
        # ノード選択用のインデックス
        self.n_selected_nodes = int(self.n_reservoir * self.node_selection_ratio)
        
        np.random.seed(self.random_state)
        self.selected_indices = np.sort(
            np.random.choice(self.n_reservoir, self.n_selected_nodes, replace=False)
        )
        
        # 分類器の選択
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
        elif classifier_type == 'ridge':
            cfg = classifier_config if classifier_config else RIDGE_CONFIG
            self.classifier = RidgeClassifier(
                alpha=cfg.get('alpha', 1.0),
                random_state=cfg.get('random_state', 42)
            )
        else:
            raise ValueError(f"Unknown classifier type: {classifier_type}")
    
    def _concatenate_all_features(self, X_md_channels, X_rtm_channels):
        """
        全チャンネル・全データタイプの時系列を特徴軸で結合
        
        Args:
            X_md_channels: {channel: [samples]} の辞書（MD/DTMデータ）
            X_rtm_channels: {channel: [samples]} の辞書（RTMデータ）
            
        Returns:
            list: 結合された時系列データのリスト [n_samples]
                  各要素は [time_frames, total_features] の配列
        """
        n_samples = len(X_md_channels[self.channels[0]])
        concatenated_sequences = []
        
        for sample_idx in range(n_samples):
            features_per_time = []
            
            # 各サンプルについて、全チャンネル・全データタイプを取得
            md_data = {ch: X_md_channels[ch][sample_idx] for ch in self.channels}
            rtm_data = {ch: X_rtm_channels[ch][sample_idx] for ch in self.channels}
            
            # 時間長は全チャンネルで同じと仮定（MD ch0 を基準）
            time_frames = md_data[self.channels[0]].shape[0]
            
            # 各時刻で全特徴を結合
            for t in range(time_frames):
                time_features = []
                
                # 各チャンネルのMDとRTMを順に結合
                for ch in self.channels:
                    time_features.append(md_data[ch][t, :])  # MD特徴
                    time_features.append(rtm_data[ch][t, :])  # RTM特徴
                
                # 横方向に結合
                features_per_time.append(np.concatenate(time_features))
            
            # [time_frames, total_features] の配列にする
            concatenated_sequences.append(np.array(features_per_time))
        
        return concatenated_sequences
    
    def _extract_features(self, X_md_channels, X_rtm_channels, verbose=False):
        """
        全チャンネル・全データタイプから特徴抽出
        
        Args:
            X_md_channels: {channel: [samples]} の辞書
            X_rtm_channels: {channel: [samples]} の辞書
            verbose: 進捗表示
            
        Returns:
            selected_features: [n_samples, n_selected_nodes]
        """
        # 全特徴を時系列方向で結合
        concatenated_sequences = self._concatenate_all_features(X_md_channels, X_rtm_channels)
        
        if verbose:
            print(f"  Processing {len(concatenated_sequences)} sequences with single reservoir...")
            print(f"    Reservoir nodes: {self.n_reservoir}")
            print(f"    Selected nodes: {self.n_selected_nodes}")
            if concatenated_sequences:
                print(f"    Input feature dimension: {concatenated_sequences[0].shape[1]}")
        
        # 単一リザバーで特徴抽出
        features = self.esn.transform_sequences(concatenated_sequences)
        
        # ノード選択
        selected_features = features[:, self.selected_indices]
        
        if verbose:
            print(f"  Selected features shape: {selected_features.shape}")
        
        return selected_features
    
    def extract_features(self, X_md_channels, X_rtm_channels, verbose=False):
        """特徴抽出のみを行う（分類器に依存しない）"""
        return self._extract_features(X_md_channels, X_rtm_channels, verbose=verbose)
    
    def fit(self, X_md_channels, X_rtm_channels, y, verbose=False):
        """
        訓練
        
        Returns:
            (feature_time, train_time): 各処理の時間
        """
        start_time = time.time()
        features = self._extract_features(X_md_channels, X_rtm_channels, verbose=verbose)
        feature_time = time.time() - start_time
        
        start_time = time.time()
        self.classifier.fit(features, y)
        train_time = time.time() - start_time
        
        if verbose:
            print(f"特徴抽出時間: {feature_time:.4f}秒")
            print(f"分類器学習時間: {train_time:.4f}秒")
        
        return feature_time, train_time
    
    def fit_from_features(self, features, y, return_breakdown=False):
        """抽出済みの特徴から訓練（高速化用）"""
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
        features = self._extract_features(X_md_channels, X_rtm_channels, verbose=verbose)
        feature_time = time.time() - start_time
        
        start_time = time.time()
        predictions = self.classifier.predict(features)
        predict_time = time.time() - start_time
        
        return predictions, feature_time, predict_time
    
    def predict_from_features(self, features, return_breakdown=False):
        """抽出済みの特徴から予測（高速化用）"""
        start_time = time.time()
        predictions = self.classifier.predict(features)
        predict_time = time.time() - start_time
        
        if return_breakdown:
            return predictions, predict_time
        return predictions
