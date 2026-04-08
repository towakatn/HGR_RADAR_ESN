"""
Reservoir Computing System
リザバーコンピューティングによる分類システム
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
import time
from typing import Tuple, Optional, Dict, Any
from tqdm import tqdm

class ReservoirComputer:
    """
    Echo State Network (ESN) を使用したリザバーコンピューティングシステム
    """
    
    def __init__(self, 
                 n_reservoir: int = 800,
                 spectral_radius: float = 0.95,
                 input_scaling: float = 0.4,
                 density: float = 0.05,
                 leakage_rate: float = 0.1,
                 random_state: Optional[int] = None):
        """
        リザバーコンピューターの初期化
        
        Args:
            n_reservoir (int): リザバーノード数
            spectral_radius (float): スペクトル半径
            input_scaling (float): 入力スケール
            density (float): 結合密度
            leakage_rate (float): リーク率
            random_state (int, optional): 乱数シード
        """
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.density = density
        self.leakage_rate = leakage_rate
        self.random_state = random_state
        
        # 再現性のためにローカルな乱数生成器を使用
        self.rng = np.random.RandomState(random_state)
        
        # リザバー重み行列とその他の重みは後で初期化
        self.W_reservoir = None
        self.W_input = None
        self.n_inputs = None
        self.states = None
        
        # 統計情報
        self.initialization_time = 0
        self.training_time = 0
        self.prediction_time = 0
    
    def _initialize_reservoir(self, n_inputs: int):
        """
        リザバー重み行列を初期化
        
        Args:
            n_inputs (int): 入力次元数
        """
        start_time = time.time()
        
        self.n_inputs = n_inputs
        
        # リザバー内部重み行列 (sparse)
        n_connections = int(self.density * self.n_reservoir * self.n_reservoir)
        
        # スパース行列でリザバー重みを生成（ローカル乱数生成器を使用）
        row_indices = self.rng.randint(0, self.n_reservoir, n_connections)
        col_indices = self.rng.randint(0, self.n_reservoir, n_connections)
        data = self.rng.uniform(-1, 1, n_connections)
        
        self.W_reservoir = sparse.csr_matrix(
            (data, (row_indices, col_indices)), 
            shape=(self.n_reservoir, self.n_reservoir)
        )
        
        # スペクトル半径を調整
        # より決定論的な振る舞いのため、可能であれば密行列に変換して numpy の固有値計算を使う。
        # 失敗した場合のみ ARPACK にフォールバックする。
        try:
            dense_W = self.W_reservoir.toarray() if sparse.issparse(self.W_reservoir) else np.array(self.W_reservoir)
            eigenvalues = np.linalg.eigvals(dense_W)
            current_spectral_radius = np.max(np.abs(eigenvalues))
        except Exception:
            eigenvalues = linalg.eigs(self.W_reservoir, k=1, which='LM', return_eigenvectors=False)
            current_spectral_radius = np.abs(eigenvalues[0])
        if current_spectral_radius > 0:
            self.W_reservoir = self.W_reservoir * (self.spectral_radius / current_spectral_radius)
        
        # 入力重み行列（ローカル乱数生成器を使用）
        self.W_input = self.rng.uniform(
            -self.input_scaling, 
            self.input_scaling, 
            (self.n_reservoir, n_inputs)
        )
        
        self.initialization_time = time.time() - start_time
    
    def _run_reservoir(self, inputs: np.ndarray, initial_state: Optional[np.ndarray] = None) -> np.ndarray:
        """
        リザバーを実行して状態を取得
        
        Args:
            inputs (np.ndarray): 入力データ (time_steps, n_inputs)
            initial_state (np.ndarray, optional): 初期状態
            
        Returns:
            np.ndarray: リザバー状態 (time_steps, n_reservoir)
        """
        n_time_steps = inputs.shape[0]
        
        # 状態初期化
        if initial_state is None:
            state = np.zeros(self.n_reservoir)
        else:
            state = initial_state.copy()
        
        states = np.zeros((n_time_steps, self.n_reservoir))
        
        # リザバー実行
        for t in range(n_time_steps):
            # 新しい状態を計算
            input_activation = np.dot(self.W_input, inputs[t])
            reservoir_activation = self.W_reservoir.dot(state)
            
            # 活性化関数: tanh
            new_state = np.tanh(input_activation + reservoir_activation)
            
            # リーキング統合
            state = (1 - self.leakage_rate) * state + self.leakage_rate * new_state
            
            states[t] = state
        
        return states
    
    def fit(self, X: list, y: np.ndarray) -> 'ReservoirComputer':
        """
        リザバーコンピューターを訓練
        
        Args:
            X (list): 入力データのリスト（各要素は(time_steps, n_features)の配列）
            y (np.ndarray): ラベル (n_samples,)
            
        Returns:
            ReservoirComputer: 訓練済みのself
        """
        start_time = time.time()
        
        # 最初のサンプルからリザバーを初期化
        if self.W_reservoir is None:
            n_inputs = X[0].shape[1]  # 特徴量次元
            self._initialize_reservoir(n_inputs)
        
        # 全サンプルでリザバー状態を計算
        all_states = []
        for i, sample_input in enumerate(tqdm(X, desc="状態計算", disable=len(X) < 100)):
            states = self._run_reservoir(sample_input)  # (time_steps, features)
            # 最後の状態のみを使用
            final_state = states[-1]  # 最終状態
            all_states.append(final_state)
        
        self.states = np.array(all_states)  # (n_samples, n_reservoir)
        self.labels = y
        
        self.training_time = time.time() - start_time
        
        return self
    
    def transform(self, X: list) -> np.ndarray:
        """
        入力データをリザバー状態に変換
        
        Args:
            X (list): 入力データのリスト（各要素は(time_steps, n_features)の配列）
            
        Returns:
            np.ndarray: リザバー状態 (n_samples, n_reservoir)
        """
        start_time = time.time()
        
        if self.W_reservoir is None:
            raise ValueError("Reservoir not initialized. Call fit() first.")
        
        all_states = []
        for sample_input in X:
            states = self._run_reservoir(sample_input)  # (time_steps, features)
            final_state = states[-1]  # 最終状態
            all_states.append(final_state)
        
        prediction_states = np.array(all_states)
        
        self.prediction_time = time.time() - start_time
        
        return prediction_states
    
    def get_reservoir_states(self) -> np.ndarray:
        """
        訓練時のリザバー状態を取得
        
        Returns:
            np.ndarray: リザバー状態 (n_samples, n_reservoir)
        """
        if self.states is None:
            raise ValueError("No states available. Call fit() first.")
        return self.states
    
    def get_timing_info(self) -> Dict[str, float]:
        """
        処理時間情報を取得
        
        Returns:
            dict: 処理時間情報
        """
        return {
            'initialization_time': self.initialization_time,
            'training_time': self.training_time,
            'prediction_time': self.prediction_time
        }
    
    def get_parameters(self) -> Dict[str, Any]:
        """
        リザバーパラメータを取得
        
        Returns:
            dict: パラメータ情報
        """
        return {
            'n_reservoir': self.n_reservoir,
            'spectral_radius': self.spectral_radius,
            'input_scaling': self.input_scaling,
            'density': self.density,
            'leakage_rate': self.leakage_rate,
            'random_state': self.random_state
        }


def prepare_rc_input(signals):
    """
    リザバーコンピューティング用の入力データを準備
    元の時間長を保持し、可変長入力として処理
    
    Args:
        signals (list): 信号データのリスト
        
    Returns:
        list: 準備されたデータのリスト（各要素は(time_steps, n_features)の配列）
    """
    prepared_data = []
    for signal in signals:
        freq_bins, time_steps = signal.shape
        
        # 転置して (time_steps, freq_bins) に変換
        prepared_signal = signal.T  # (time_steps, freq_bins)
        prepared_data.append(prepared_signal)
    
    return prepared_data


def main():
    """テスト実行"""
    print("Reservoir Computing System Test")
    print("="*60)
    
    # テストデータ生成
    n_samples = 10
    n_time_steps = 50
    n_features = 800
    
    X = np.random.rand(n_samples, n_time_steps, n_features)
    y = np.random.randint(0, 4, n_samples)
    
    # リザバーコンピューター初期化
    rc = ReservoirComputer(
        n_reservoir=800,
        spectral_radius=0.95,
        input_scaling=0.4,
        density=0.05,
        leakage_rate=0.1,
        random_state=42
    )
    
    # 訓練
    rc.fit(X, y)
    
    # 変換
    states = rc.transform(X)
    
    print(f"\nTest completed successfully!")
    print(f"Output states shape: {states.shape}")
    print(f"Timing info: {rc.get_timing_info()}")
    
    return rc


if __name__ == "__main__":
    rc = main()
