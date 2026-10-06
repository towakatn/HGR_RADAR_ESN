#!/usr/bin/env python3
"""Shared dataset loaders for Dop-NET MATLAB signals and Soli DTM/RTM data."""

import numpy as np
import scipy.io as sio
import os
import time
from typing import Tuple, List, Dict, Any
import h5py
import glob

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



class DualDataTypeLoader:
    """MDDataとRTMData両方を4チャンネルから読み込むクラス"""

    def __init__(self, channels=[0, 1, 2, 3], base_dir="."):
        """
        Args:
            channels: 使用するチャンネルリスト (デフォルト: [0, 1, 2, 3])
            base_dir: ベースディレクトリ（デフォルト: カレントディレクトリ）
        """
        for ch in channels:
            if ch not in [0, 1, 2, 3]:
                raise ValueError(f"Channel must be 0, 1, 2, or 3. Got: {ch}")

        self.channels = channels
        self.base_dir = base_dir
        self.md_dirs = {ch: os.path.join(base_dir, "DTM", f"{ch}ch_DTMData") for ch in channels}
        self.rtm_dirs = {ch: os.path.join(base_dir, "RTM", f"{ch}ch_RTMData") for ch in channels}
        self.gesture_names = {
            0: "Pinch Index",
            1: "Pinch Pinky",
            2: "Finger Slide",
            3: "Finger Rub",
            4: "Slow Swipe",
            5: "Fast Swipe",
            6: "Push",
            7: "Pull",
            8: "Palm Tilt",
            9: "Circle",
            10: "Palm Hold"
        }

    def load_gesture_data(self, max_samples_per_gesture_subject=25):
        """
        MDDataとRTMData両方を4チャンネル全てから読み込み

        Args:
            max_samples_per_gesture_subject: 各ジェスチャー・被験者の組み合わせで使用する最大サンプル数

        Returns:
            X_md_channels: 辞書 {channel: リスト of [time_frames, doppler_bins]}
            X_rtm_channels: 辞書 {channel: リスト of [time_frames, range_bins]}
            y: ラベル配列
            metadata: メタデータリスト
        """
        X_md_channels = {ch: [] for ch in self.channels}
        X_rtm_channels = {ch: [] for ch in self.channels}
        y = []
        filenames = []

        for ch in self.channels:
            if not os.path.exists(self.md_dirs[ch]):
                raise FileNotFoundError(f"MD data directory not found: {self.md_dirs[ch]}")
            if not os.path.exists(self.rtm_dirs[ch]):
                raise FileNotFoundError(f"RTM data directory not found: {self.rtm_dirs[ch]}")

        for gesture_class in range(11):
            pattern = f"rde_ch0_{gesture_class}_*_*.h5"
            md_files = glob.glob(os.path.join(self.md_dirs[0], pattern))

            base_files = [os.path.basename(f).replace(f"rde_ch0_", "") for f in md_files]
            base_files = sorted(base_files)

            filtered_files = []
            for base_filename in base_files:
                try:
                    parts = base_filename.replace('.h5', '').split('_')
                    if len(parts) >= 3:
                        subject_num = int(parts[1])
                        sample_num = int(parts[2])
                        if subject_num < 10 and sample_num < max_samples_per_gesture_subject:
                            filtered_files.append(base_filename)
                except (ValueError, IndexError):
                    continue

            for base_filename in filtered_files:
                try:
                    md_data = {}
                    rtm_data = {}
                    valid_sample = True

                    for ch in self.channels:
                        md_filename = f"rde_ch{ch}_{base_filename}"
                        md_path = os.path.join(self.md_dirs[ch], md_filename)

                        rtm_filename = f"rtm_ch{ch}_{base_filename}"
                        rtm_path = os.path.join(self.rtm_dirs[ch], rtm_filename)

                        if os.path.exists(md_path) and os.path.exists(rtm_path):
                            with h5py.File(md_path, 'r') as f:
                                md_data[ch] = f['rd_evolution'][:]

                            with h5py.File(rtm_path, 'r') as f:
                                rtm_data[ch] = f['rtm'][:]
                        else:
                            valid_sample = False
                            break

                    if valid_sample:
                        for ch in self.channels:
                            X_md_channels[ch].append(md_data[ch])
                            X_rtm_channels[ch].append(rtm_data[ch])
                        y.append(gesture_class)
                        filenames.append(base_filename)

                except Exception as e:
                    continue

        y = np.array(y)

        metadata = []
        for filename in filenames:
            parts = filename.replace('.h5', '').split('_')
            if len(parts) >= 3:
                gesture = int(parts[0])
                subject = int(parts[1])
                session = int(parts[2])
                metadata.append({
                    'gesture': gesture,
                    'subject': subject,
                    'session': session,
                    'filename': filename
                })

        return X_md_channels, X_rtm_channels, y, metadata

