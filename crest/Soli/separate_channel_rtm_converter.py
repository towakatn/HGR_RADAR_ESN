#!/usr/bin/env python3

import h5py
import numpy as np
import os
import glob
from tqdm import tqdm
import json
import datetime

class SeparateChannelRTMConverter:
    def __init__(self, input_dir="SoliData/dsp", base_output_dir="RTM"):
        self.input_dir = input_dir
        self.base_output_dir = base_output_dir
        self.sampling_rate = 40
        
        if not os.path.exists(self.base_output_dir):
            os.makedirs(self.base_output_dir)
        
        self.channel_dirs = {}
        for ch in range(4):
            output_dir = os.path.join(self.base_output_dir, f"{ch}ch_RTMData")
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
    
    def extract_range_time_map(self, rd_data):
        try:
            if rd_data.ndim == 2 and rd_data.shape[1] == 1024:
                frames = rd_data.shape[0]
                rd_reshaped = rd_data.reshape(frames, 32, 32)
            elif rd_data.ndim == 3:
                rd_reshaped = rd_data
            else:
                return None, None, None
            
            frames, range_bins, doppler_bins = rd_reshaped.shape
            rtm = np.sum(rd_reshaped, axis=2)
            time_axis = np.linspace(0, frames / self.sampling_rate, frames)
            range_axis = np.arange(range_bins)
            
            return rtm, time_axis, range_axis
            
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
                    rtm, time_axis, range_axis = self.extract_range_time_map(rd_data)
                    
                    if rtm is None:
                        results[ch] = None
                        continue
                    
                    output_filename = f"rtm_ch{ch}_{os.path.splitext(filename)[0]}.h5"
                    output_path = os.path.join(self.channel_dirs[ch], output_filename)
                    
                    with h5py.File(output_path, 'w') as f_out:
                        f_out.create_dataset('rtm', data=rtm)
                        f_out.create_dataset('time_axis', data=time_axis)
                        f_out.create_dataset('range_axis', data=range_axis)
                        
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
                        f_out.attrs['data_type'] = 'range_time_map'
                        f_out.attrs['description'] = f'Channel {ch} Doppler-integrated Range evolution over time'
                        f_out.attrs['shape_info'] = f"Original: {rd_data.shape}, RTM: {rtm.shape}"
                    
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
                'conversion_type': 'range_time_map',
                'channel': ch,
                'input_directory': self.input_dir,
                'output_directory': self.channel_dirs[ch],
                'total_input_files': len(input_files),
                'successful_conversions': len(channel_stats[ch]['successful_conversions']),
                'failed_conversions': len(channel_stats[ch]['failed_conversions']),
                'success_rate': len(channel_stats[ch]['successful_conversions']) / len(input_files) * 100 if input_files else 0,
                'failed_files': [os.path.basename(f) for f in channel_stats[ch]['failed_conversions']]
            }
            
            log_filename = f"conversion_log_ch{ch}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            log_path = os.path.join(self.channel_dirs[ch], "logs", log_filename)
            
            with open(log_path, 'w') as log_file:
                json.dump(log_data, log_file, indent=2)

def main():
    converter = SeparateChannelRTMConverter()
    converter.convert_all_files()

if __name__ == "__main__":
    main()
