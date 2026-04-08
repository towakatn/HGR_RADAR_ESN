#!/usr/bin/env python3
"""
データローダーモジュール
MDData（Doppler-Time）とRTMData（Range-Time）のデュアルデータタイプを読み込む
"""

import numpy as np
import h5py
import os
import glob


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
