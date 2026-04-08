#!/usr/bin/env python3
"""
Feature-Based ESN Readout
非線形拡張特徴とRidge回帰によるリードアウト
"""

import numpy as np
import time
from sklearn.preprocessing import OneHotEncoder
from .reservoir import VariableLengthESN


class FeatESNReadout:
    """
    Feature-Based ESN Readout
    
    マルチリザバー構造:
    - MDData（Doppler-Time）4チャンネル × RTMData（Range-Time）4チャンネル = 8リザバー
    - 各リザバーの最終状態を統合し、非線形拡張特徴を構成
    - Ridge回帰によりリードアウト重みを学習
    
    非線形拡張特徴:
    - 'none': Ψ(r) = r のみ（RR_L用）
    - 'square_tanh': Ψ(r) = [1, r^T, tanh(r)^T]^T（RR_N用）
    """
    
    def __init__(self, n_reservoir_per_stream=50, n_selected_nodes=50,
                 spectral_radius=0.95, input_scaling=1.0, density=0.9,
                 leakage_rate=0.0263, bias_scaling=0.0,
                 regularization=0.001,
                 nonlinear_features='square_tanh',
                 random_state=42):
        """
        Feat-ESN Readout の初期化
        
        Args:
            n_reservoir_per_stream: 各ストリームのリザバーノード数
            n_selected_nodes: 各ストリームから選択するノード数
            spectral_radius: スペクトル半径
            input_scaling: 入力スケーリング
            density: リザバー接続密度
            leakage_rate: リーク率
            bias_scaling: バイアススケーリング
            regularization: Tikhonov正則化係数 (λ)
            nonlinear_features: 'none' or 'square_tanh'
            random_state: 乱数シード
        """
        self.n_reservoir_per_stream = n_reservoir_per_stream
        self.n_selected_nodes = n_selected_nodes
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.density = density
        self.leakage_rate = leakage_rate
        self.bias_scaling = bias_scaling
        self.regularization = regularization
        self.nonlinear_features = nonlinear_features
        self.random_state = random_state
        
        self.esns_md = {}
        self.esns_rtm = {}
        
        for ch in range(4):
            self.esns_md[ch] = VariableLengthESN(
                n_reservoir=n_reservoir_per_stream,
                spectral_radius=spectral_radius,
                input_scaling=input_scaling,
                density=density,
                leakage_rate=leakage_rate,
                bias_scaling=bias_scaling,
                random_state=random_state + ch
            )
            
            self.esns_rtm[ch] = VariableLengthESN(
                n_reservoir=n_reservoir_per_stream,
                spectral_radius=spectral_radius,
                input_scaling=input_scaling,
                density=density,
                leakage_rate=leakage_rate,
                bias_scaling=bias_scaling,
                random_state=random_state + ch + 100
            )
        
        np.random.seed(random_state)
        self.selected_nodes_md = {}
        self.selected_nodes_rtm = {}
        for ch in range(4):
            self.selected_nodes_md[ch] = sorted(
                np.random.choice(n_reservoir_per_stream, self.n_selected_nodes, replace=False)
            )
            self.selected_nodes_rtm[ch] = sorted(
                np.random.choice(n_reservoir_per_stream, self.n_selected_nodes, replace=False)
            )
        
        self.W_out = None
        self.n_classes = None
        self.label_encoder = None
    
    def _extract_reservoir_states(self, X_md_channels, X_rtm_channels, verbose=False):
        """
        全チャンネル・全データタイプからリザバー状態を抽出して統合
        
        Returns:
            r: 統合リザバー状態 [n_samples, total_selected_nodes]
        """
        all_states = []
        
        for ch in range(4):
            states = self.esns_md[ch].transform_sequences(X_md_channels[ch])
            selected = states[:, self.selected_nodes_md[ch]]
            all_states.append(selected)
            if verbose:
                print(f"  MD Ch{ch}: {selected.shape}")
        
        for ch in range(4):
            states = self.esns_rtm[ch].transform_sequences(X_rtm_channels[ch])
            selected = states[:, self.selected_nodes_rtm[ch]]
            all_states.append(selected)
            if verbose:
                print(f"  RTM Ch{ch}: {selected.shape}")
        
        r = np.hstack(all_states)
        
        if verbose:
            print(f"  統合リザバー状態 r: {r.shape}")
        
        return r
    
    def _construct_extended_features(self, r):
        """
        非線形拡張特徴 Ψ(r) の構成
        """
        n_samples, n_reservoir = r.shape
        
        if self.nonlinear_features == 'none':
            Psi = r
        elif self.nonlinear_features == 'square_tanh':
            bias = np.ones((n_samples, 1))
            r_tanh = np.tanh(r)
            Psi = np.hstack([bias, r, r_tanh])
        else:
            raise ValueError(f"Unknown nonlinear_features: {self.nonlinear_features}")
        
        return Psi
    
    def fit(self, X_md_channels, X_rtm_channels, y, verbose=False):
        """
        Ridge回帰によるリードアウト層の学習
        
        Returns:
            (feature_time, readout_time): 各処理の時間
        """
        start_time = time.time()
        r = self._extract_reservoir_states(X_md_channels, X_rtm_channels, verbose=verbose)
        feature_time = time.time() - start_time
        
        if verbose:
            print(f"特徴抽出時間: {feature_time:.4f}秒")
        
        start_time = time.time()
        Psi = self._construct_extended_features(r)
        
        if verbose:
            print(f"拡張特徴 Ψ(r): {Psi.shape}")
        
        self.n_classes = len(np.unique(y))
        
        self.label_encoder = OneHotEncoder(sparse_output=False, categories='auto')
        Y = self.label_encoder.fit_transform(y.reshape(-1, 1))
        
        if verbose:
            print(f"教師信号 Y: {Y.shape}")
        
        n_samples, n_features = Psi.shape
        PsiT_Psi = np.dot(Psi.T, Psi)
        lambda_I = self.regularization * np.eye(n_features)
        
        try:
            inv_matrix = np.linalg.inv(PsiT_Psi + lambda_I)
        except np.linalg.LinAlgError:
            if verbose:
                print("警告: 逆行列計算失敗、疑似逆行列を使用")
            inv_matrix = np.linalg.pinv(PsiT_Psi + lambda_I)
        
        W_temp = np.dot(inv_matrix, Psi.T)
        self.W_out = np.dot(W_temp, Y).T
        
        readout_time = time.time() - start_time
        
        if verbose:
            print(f"Readout重み W_out: {self.W_out.shape}")
            print(f"Readout学習時間: {readout_time:.4f}秒")
        
        return feature_time, readout_time
    
    def predict(self, X_md_channels, X_rtm_channels, verbose=False):
        """
        Ridge回帰による予測
        
        Returns:
            (predictions, feature_time, readout_time)
        """
        start_time = time.time()
        r = self._extract_reservoir_states(X_md_channels, X_rtm_channels, verbose=verbose)
        feature_time = time.time() - start_time
        
        start_time = time.time()
        Psi = self._construct_extended_features(r)
        
        y_scores = np.dot(Psi, self.W_out.T)
        predictions = np.argmax(y_scores, axis=1)
        
        readout_time = time.time() - start_time
        
        return predictions, feature_time, readout_time
