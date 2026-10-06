#!/usr/bin/env python3
"""共通ESNリードアウト。データセット固有の設定は呼び出し側から渡す。"""

import time
import numpy as np
from sklearn.preprocessing import OneHotEncoder

from .classifiers import create_classifier
from .reservoir import VariableLengthESN


def fit_ridge_readout(features, targets, regularization, verbose=False):
    """Fit the original RR_L/RR_N readout; return weights shaped (classes, features).

    Keep the original inverse/pseudoinverse fallback and multiplication order.
    The regularizer is lambda * I, without sample-count scaling.
    """
    n_features = features.shape[1]
    PsiT_Psi = np.dot(features.T, features)
    lambda_I = regularization * np.eye(n_features)

    try:
        inv_matrix = np.linalg.inv(PsiT_Psi + lambda_I)
    except np.linalg.LinAlgError:
        if verbose:
            print("警告: 逆行列計算失敗、疑似逆行列を使用")
        inv_matrix = np.linalg.pinv(PsiT_Psi + lambda_I)

    W_temp = np.dot(inv_matrix, features.T)
    return np.dot(W_temp, targets).T


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
                 n_reservoir=50, spectral_radius=0.95, input_scaling=1.0,
                 density=0.9, leakage_rate=0.0263, bias_scaling=0.0,
                 random_state=42,
                 classifier_config=None):
        """
        Classifier-Based ESN Readout の初期化

        Args:
            classifier_type: 'rf' or 'svm'
            n_reservoir: リザバーノード数（Noneの場合は従来のデフォルト値を使用）
            spectral_radius: スペクトル半径
            input_scaling: 入力スケーリング
            density: リザバー接続密度
            leakage_rate: リーク率
            bias_scaling: バイアススケーリング
            random_state: 乱数シード
            classifier_config: 分類器固有の設定（Noneの場合は従来のデフォルト値を使用）
        """
        # Noneが渡された場合も従来のデフォルト値を使用
        self.n_reservoir = n_reservoir if n_reservoir is not None else 50
        self.spectral_radius = spectral_radius if spectral_radius is not None else 0.95
        self.input_scaling = input_scaling if input_scaling is not None else 1.0
        self.density = density if density is not None else 0.9
        self.leakage_rate = leakage_rate if leakage_rate is not None else 0.0263
        self.bias_scaling = bias_scaling if bias_scaling is not None else 0.0
        self.random_state = random_state if random_state is not None else 42

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
            cfg = classifier_config if classifier_config else {}
            self.classifier = create_classifier('rf', dict(
                n_estimators=cfg.get('n_estimators', 300),
                max_depth=cfg.get('max_depth', None),
                random_state=cfg.get('random_state', 42),
                n_jobs=cfg.get('n_jobs', -1)
            ))
        elif classifier_type == 'svm':
            cfg = classifier_config if classifier_config else {}
            self.classifier = create_classifier('svm', dict(
                kernel=cfg.get('kernel', 'rbf'),
                C=cfg.get('C', 10.0),
                gamma=cfg.get('gamma', 'scale'),
                random_state=cfg.get('random_state', 42)
            ))
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

        self.W_out = fit_ridge_readout(Psi, Y, self.regularization, verbose=verbose)

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


class SingleReservoirESN:
    """
    全チャンネル・全データタイプを結合して1つのリザバーで処理するESNクラス
    """

    def __init__(self, channels=[0, 1, 2, 3],
                 n_reservoir=500, spectral_radius=0.95, input_scaling=0.2,
                 density=0.1, leakage_rate=0.05, bias_scaling=0.0,
                 node_selection_ratio=1.0,
                 classifier_type='rf', random_state=42,
                 classifier_config=None):
        """
        Args:
            channels: 使用するチャンネル
            n_reservoir: リザバーノード数（Noneの場合は従来のデフォルト値を使用）
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

        # Noneが渡された場合も従来のデフォルト値を使用
        self.n_reservoir = n_reservoir if n_reservoir is not None else 500
        self.spectral_radius = spectral_radius if spectral_radius is not None else 0.95
        self.input_scaling = input_scaling if input_scaling is not None else 0.2
        self.density = density if density is not None else 0.1
        self.leakage_rate = leakage_rate if leakage_rate is not None else 0.05
        self.bias_scaling = bias_scaling if bias_scaling is not None else 0.0
        self.node_selection_ratio = node_selection_ratio if node_selection_ratio is not None else 1.0
        self.random_state = random_state if random_state is not None else 42

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
            cfg = classifier_config if classifier_config else {}
            self.classifier = create_classifier('rf', dict(
                n_estimators=cfg.get('n_estimators', 300),
                max_depth=cfg.get('max_depth', None),
                random_state=cfg.get('random_state', 42),
                n_jobs=cfg.get('n_jobs', -1)
            ))
        elif classifier_type == 'svm':
            cfg = classifier_config if classifier_config else {}
            self.classifier = create_classifier('svm', dict(
                kernel=cfg.get('kernel', 'rbf'),
                C=cfg.get('C', 10.0),
                gamma=cfg.get('gamma', 'scale'),
                random_state=cfg.get('random_state', 42)
            ))
        elif classifier_type == 'ridge':
            cfg = classifier_config if classifier_config else {}
            self.classifier = create_classifier('ridge', dict(
                alpha=cfg.get('alpha', 1.0),
                random_state=cfg.get('random_state', 42)
            ))
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
