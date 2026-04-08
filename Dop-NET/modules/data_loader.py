"""
Reservoir Computing用データローダー
時間正規化なし、ログ変換なしの正規化データを読み込み
"""

import numpy as np
import scipy.io as sio
import os
import time
from typing import Tuple, List, Dict, Any

class RCDataLoader:
    """
    Reservoir Computing用のデータローダー
    時間正規化なし、ログ変換なしで正規化データを提供
    """
    
    def __init__(self, data_dir: str = "Data/Training Data"):
        """
        初期化
        
        Args:
            data_dir (str): MATLABファイルが格納されているディレクトリ
        """
        self.data_dir = data_dir
        self.persons = ['A', 'B', 'C', 'D', 'E', 'F']
        # Label order: 0=Wave, 1=Pinch, 2=Swipe, 3=Click
        self.gestures = ['Wave', 'Pinch', 'Swipe', 'Click']
        self.gesture_mapping = {gesture: idx for idx, gesture in enumerate(self.gestures)}
        
        self.processing_times = []
        self.original_lengths = []
        
    def convert_to_normalized_spectrogram(self, doppler_signal):
        """
        ドップラー信号を正規化された振幅スペクトログラムに変換
        ログ変換は行わず、[0,1]の範囲で正規化のみ
        
        Args:
            doppler_signal (numpy.ndarray): 複素ドップラー信号
            
        Returns:
            numpy.ndarray: 振幅スペクトログラム [0,1]
        """
        # 振幅スペクトログラム
        abs_signal = np.abs(doppler_signal)
        
        
        return abs_signal
    
    def load_single_file(self, person: str) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """
        単一人物のMATLABファイルを読み込み
        
        Args:
            person (str): 人物ID ('A', 'B', 'C', 'D', 'E', 'F')
            
        Returns:
            tuple: (signals, labels, metadata)
        """
        filename = f"Data_Per_PersonData_Training_Person_{person}.mat"
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")
        
        # MATLABファイル読み込み
        mat_data = sio.loadmat(filepath)
        doppler_signals = mat_data["Data_Training"]["Doppler_Signals"][0][0][0]
        
        signals = []
        labels = []
        metadata = []
        
        for gesture_idx in range(4):  # 0: Wave, 1: Pinch, 2: Swipe, 3: Click
            gesture_data = doppler_signals[gesture_idx]
            
            # 各ジェスチャーの全サンプルを処理
            for sample_idx in range(len(gesture_data)):
                try:
                    start_time = time.time()
                    
                    # ドップラー信号取得
                    doppler_signal = gesture_data[sample_idx][0]
                    
                    # 信号が有効かチェック
                    if doppler_signal is None or doppler_signal.size == 0:
                        continue
                    
                    # 元の長さを記録
                    original_length = doppler_signal.shape[1] if len(doppler_signal.shape) > 1 else len(doppler_signal)
                    self.original_lengths.append(original_length)
                    
                    # 正規化スペクトログラムに変換（ログ変換なし）
                    normalized_signal = self.convert_to_normalized_spectrogram(doppler_signal)
                    
                    # データ保存
                    signals.append(normalized_signal)
                    labels.append(gesture_idx)
                    metadata.append({
                        'person': person,
                        'gesture': self.gestures[gesture_idx],
                        'sample_idx': sample_idx,
                        'original_length': original_length
                    })
                    
                    # 処理時間記録
                    processing_time = time.time() - start_time
                    self.processing_times.append(processing_time)
                    
                except (IndexError, AttributeError) as e:
                    continue
        
        return signals, labels, metadata
    
    def load_all_data(self) -> Tuple[List[np.ndarray], List[int], List[Dict]]:
        """
        全ての人物のデータを読み込み
        
        Returns:
            tuple: (all_signals, all_labels, all_metadata)
        """
        all_signals = []
        all_labels = []
        all_metadata = []
        
        for person in self.persons:
            signals, labels, metadata = self.load_single_file(person)
            all_signals.extend(signals)
            all_labels.extend(labels)
            all_metadata.extend(metadata)
        
        return all_signals, all_labels, all_metadata
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        データロード統計情報を取得
        
        Returns:
            dict: 統計情報
        """
        if not self.processing_times:
            return {}
        
        return {
            'total_samples': len(self.processing_times),
            'processing_times': {
                'total': sum(self.processing_times),
                'mean': np.mean(self.processing_times),
                'std': np.std(self.processing_times),
                'min': min(self.processing_times),
                'max': max(self.processing_times)
            },
            'original_lengths': {
                'min': min(self.original_lengths),
                'max': max(self.original_lengths),
                'mean': np.mean(self.original_lengths),
                'median': np.median(self.original_lengths),
                'std': np.std(self.original_lengths)
            }
        }
    
    def print_statistics(self):
        """統計情報を表示"""
        stats = self.get_statistics()
        if not stats:
            print("No statistics available")
            return
        
        print("\n" + "="*80)
        print("RC DATA LOADING STATISTICS")
        print("="*80)
        
        print(f"Total Samples Processed: {stats['total_samples']}")
        print(f"Data Format: Normalized amplitude [0,1] - NO log transform, NO temporal normalization")
        
        print(f"\nOriginal Length Distribution:")
        print(f"  Min:    {stats['original_lengths']['min']}")
        print(f"  Max:    {stats['original_lengths']['max']}")
        print(f"  Mean:   {stats['original_lengths']['mean']:.2f}")
        print(f"  Median: {stats['original_lengths']['median']:.2f}")
        print(f"  Std:    {stats['original_lengths']['std']:.2f}")
        
        print(f"\nProcessing Time per Sample:")
        print(f"  Total:  {stats['processing_times']['total']:.3f} seconds")
        print(f"  Mean:   {stats['processing_times']['mean']*1000:.3f} ms")
        print(f"  Std:    {stats['processing_times']['std']*1000:.3f} ms")
        print(f"  Min:    {stats['processing_times']['min']*1000:.3f} ms")
        print(f"  Max:    {stats['processing_times']['max']*1000:.3f} ms")
        
        print("="*80)


def main():
    """テスト実行"""
    loader = RCDataLoader()
    
    # データ読み込み
    signals, labels, metadata = loader.load_all_data()
    
    # 統計表示
    loader.print_statistics()
    
    # データ形状確認
    print(f"\nData Shape Information:")
    print(f"Total samples: {len(signals)}")
    print(f"Sample shape example: {signals[0].shape}")
    print(f"Labels: {set(labels)}")
    print(f"Data type: {signals[0].dtype}")
    
    # 値の範囲確認
    sample_data = signals[0]
    print(f"\nValue Range Check (First sample):")
    print(f"  Min: {np.min(sample_data):.6f}")
    print(f"  Max: {np.max(sample_data):.6f}")
    print(f"  Mean: {np.mean(sample_data):.6f}")
    print(f"  Data format: {'[0,1] normalized' if 0 <= np.min(sample_data) and np.max(sample_data) <= 1 else 'Unknown'}")
    
    return signals, labels, metadata


if __name__ == "__main__":
    signals, labels, metadata = main()
