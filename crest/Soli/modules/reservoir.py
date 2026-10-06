#!/usr/bin/env python3
"""
Echo State Network (ESN) リザバーコンポーネント
可変長時系列データに対応したリザバーコンピューティングの実装
"""

import numpy as np


class VariableLengthESN:
    """可変長データに対応したEcho State Network"""
    
    def __init__(self, n_reservoir=50, spectral_radius=0.95, input_scaling=1.0, 
                 density=0.9, leakage_rate=0.0263, bias_scaling=0.0, random_state=42):
        """
        ESNの初期化
        
        Args:
            n_reservoir: リザバーノード数
            spectral_radius: スペクトル半径
            input_scaling: 入力スケーリング
            density: リザバー接続密度
            leakage_rate: リーク率 (α)
            bias_scaling: バイアススケーリング
            random_state: 乱数シード
        """
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.density = density
        self.leakage_rate = leakage_rate
        self.bias_scaling = bias_scaling
        self.random_state = random_state
        
        self.W_in = None
        self.W_res = None
        self.W_bias = None
        
        np.random.seed(random_state)
    
    def _initialize_weights(self, n_inputs):
        """重み行列の初期化"""
        scale = np.sqrt(2.0 / (n_inputs + self.n_reservoir))
        self.W_in = np.random.normal(0, scale, (self.n_reservoir, n_inputs))
        self.W_bias = np.random.uniform(-self.bias_scaling, self.bias_scaling, self.n_reservoir)
        self.W_res = np.random.normal(0, 1, (self.n_reservoir, self.n_reservoir))
        
        mask = np.random.rand(self.n_reservoir, self.n_reservoir) < self.density
        self.W_res *= mask
        
        eigenvalues = np.linalg.eigvals(self.W_res)
        current_spectral_radius = np.max(np.abs(eigenvalues))
        if current_spectral_radius > 0:
            self.W_res *= (self.spectral_radius / current_spectral_radius)
    
    def transform_sequences(self, X_list):
        """
        可変長時系列データをリザバー状態に変換
        
        Args:
            X_list: 時系列データのリスト [n_samples]
                    各要素は (time_steps, n_features) の配列
        
        Returns:
            states: 最終状態の配列 [n_samples, n_reservoir]
        """
        n_samples = len(X_list)
        states = np.zeros((n_samples, self.n_reservoir))
        
        for i, X in enumerate(X_list):
            time_steps, n_features = X.shape
            
            if self.W_in is None:
                self._initialize_weights(n_features)
            
            state = np.zeros(self.n_reservoir)
            
            for t in range(time_steps):
                u = X[t]
                pre_activation = (np.dot(self.W_in, u) + 
                                np.dot(self.W_res, state) + 
                                self.W_bias)
                new_state = np.tanh(pre_activation)
                state = (1 - self.leakage_rate) * state + self.leakage_rate * new_state
            
            states[i] = state
        
        return states
