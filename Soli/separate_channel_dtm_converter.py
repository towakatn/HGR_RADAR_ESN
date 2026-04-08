#!/usr/bin/env python3

import h5py
import numpy as np
import os
import glob
from tqdm import tqdm
import json
import datetime

class SeparateChannelConverter:
    def __init__(self, input_dir="SoliData/dsp", base_output_dir="DTM"):
        self.input_dir = input_dir
        self.base_output_dir = base_output_dir
        self.sampling_rate = 40
        
        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)
        
        self.channel_dirs = {}
        for ch in range(4):
            output_dir = os.path.join(self.base_output_dir, f"{ch}ch_DTMData")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            self.channel_dirs[ch] = output_dir
            
            logs_dir = os.path.join(output_dir, "logs")
            if not os.path.exists(logs_dir):
                os.makedirs(logs_dir)
    
    def scan_input_files(self):
        if not os.path.exists(self.input_dir):
            return []
        
        files = glob.glob(os.path.join(self.input_dir, "*.h5"))
        return files
    
    def extract_range_doppler_evolution(self, rd_data):
        try:
            if rd_data.ndim == 2 and rd_data.shape[1] == 1024:
                frames = rd_data.shape[0]
                rd_reshaped = rd_data.reshape(frames, 32, 32)
            elif rd_data.ndim == 3:
                rd_reshaped = rd_data
            else:
                return None, None, None
            
            frames, range_bins, doppler_bins = rd_reshaped.shape
            rd_evolution = np.sum(rd_reshaped, axis=1)
            time_axis = np.linspace(0, frames / self.sampling_rate, frames)
            doppler_axis = np.arange(doppler_bins) - doppler_bins // 2
            
            return rd_evolution, time_axis, doppler_axis
            
        except Exception as e:
            return None, None, None
    
    def process_single_file(self, file_path):
        filename = os.path.basename(file_path)
        results = {}
        
        try:
            with h5py.File(file_path, 'r') as f:
                metadata = {}
                if 'label' in f:
                    metadata['label'] = f['label'][...]
                if 'timestamp' in f:
                    metadata['timestamp'] = f['timestamp'][...]
                
                for ch in range(4):
                    ch_key = f'ch{ch}'
                    
                    if ch_key not in f:
                        results[ch] = None
                        continue
                    
                    rd_data = f[ch_key][...]
                    rd_evolution, time_axis, doppler_axis = self.extract_range_doppler_evolution(rd_data)
                    
                    if rd_evolution is None:
                        results[ch] = None
                        continue
                    
                    output_filename = f"rde_ch{ch}_{os.path.splitext(filename)[0]}.h5"
                    output_path = os.path.join(self.channel_dirs[ch], output_filename)
                    
                    with h5py.File(output_path, 'w') as f_out:
                        f_out.create_dataset('rd_evolution', data=rd_evolution)
                        f_out.create_dataset('time_axis', data=time_axis)
                        f_out.create_dataset('doppler_axis', data=doppler_axis)
                        
                        if metadata:
                            metadata_group = f_out.create_group('metadata')
                            for key, value in metadata.items():
                                try:
                                    metadata_group.create_dataset(key, data=value)
                                except:
                                    metadata_group.attrs[key] = str(value)
                        
                        f_out.attrs['source_file'] = filename
                        f_out.attrs['channel'] = ch
                        f_out.attrs['conversion_time'] = datetime.datetime.now().isoformat()
                        f_out.attrs['sampling_rate'] = self.sampling_rate
                        f_out.attrs['data_type'] = 'range_doppler_time_evolution'
                        f_out.attrs['description'] = f'Channel {ch} Range-integrated Doppler evolution over time'
                        f_out.attrs['shape_info'] = f"Original: {rd_data.shape}, Evolution: {rd_evolution.shape}"
                    
                    results[ch] = output_path
        
        except Exception as e:
            for ch in range(4):
                results[ch] = None
        
        return results
    
    def convert_all_files(self):
        input_files = self.scan_input_files()
        
        if not input_files:
            return
        
        channel_stats = {}
        for ch in range(4):
            channel_stats[ch] = {
                'successful_conversions': [],
                'failed_conversions': []
            }
        
        for file_path in tqdm(input_files, desc="Converting", ncols=80, leave=True):
            results = self.process_single_file(file_path)
            
            for ch in range(4):
                if results[ch]:
                    channel_stats[ch]['successful_conversions'].append((file_path, results[ch]))
                else:
                    channel_stats[ch]['failed_conversions'].append(file_path)
        
        for ch in range(4):
            log_data = {
                'conversion_time': datetime.datetime.now().isoformat(),
                'conversion_type': 'range_doppler_time_evolution',
                'channel': ch,
                'input_directory': self.input_dir,
                'output_directory': self.channel_dirs[ch],
                'total_files': len(input_files),
                'successful_conversions': len(channel_stats[ch]['successful_conversions']),
                'failed_conversions': len(channel_stats[ch]['failed_conversions']),
                'success_rate': len(channel_stats[ch]['successful_conversions']) / len(input_files) * 100 if input_files else 0,
                'successful_files': [{'input': inp, 'output': out} for inp, out in channel_stats[ch]['successful_conversions']],
                'failed_files': channel_stats[ch]['failed_conversions'],
                'data_description': {
                    'type': f'Channel {ch} Range-Doppler Time Evolution',
                    'format': '[time_frames, doppler_bins]',
                    'processing': 'Sum over range bins for each time frame',
                    'axes': {
                        'x_axis': 'Time (seconds)',
                        'y_axis': 'Doppler bins (centered around 0)'
                    }
                }
            }
            
            logs_dir = os.path.join(self.channel_dirs[ch], "logs")
            log_file = os.path.join(logs_dir, f"rd_evolution_ch{ch}_conversion_log_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(log_file, 'w') as f:
                json.dump(log_data, f, indent=2)
        
        return channel_stats

def main():
    converter = SeparateChannelConverter()
    channel_stats = converter.convert_all_files()

if __name__ == "__main__":
    main()
