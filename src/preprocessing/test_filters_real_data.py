#!/usr/bin/env python3
"""
Visual comparison test using real HBN data.
Direct approach without complex import handling.
"""

import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os

sys.path.insert(0, '/home/mts/HBN_EEG_Analysis')
# Import our filtering functions FIRST (while we're in the right directory)
from src.preprocessing.filters import (
    apply_notch_filter, apply_bandpass_filter, apply_average_reference,
    apply_custom_notch, apply_custom_scipy_filter, apply_custom_reference
)


from src.loader import SimpleHBNLoader, SimpleConfig

SUBJECT = "NDARAC904DMU"
TASK = "RestingState"
CHANNEL = "E45" # channels E1 to E128 (+ Cz)
TIME_WINDOW = 10

def load_real_eeg_data():
    """
    Load real EEG data by running the existing test_loader.py script.
    
    Returns:
        mne.io.Raw: Real EEG data or None if failed
    """
    try:

        config = SimpleConfig(data_root=Path("/home/mts/EFG2025_HBN-EEG_databse/data"))
        loader = SimpleHBNLoader(config)
        
        data = loader.get_data(SUBJECT, TASK)
        raw = data['raw']
        
        print(f" Loaded real EEG data: Subject {SUBJECT}, Task {TASK}")
        print(f" Data shape: {raw.get_data().shape}")
        print(f" Sampling rate: {raw.info['sfreq']} Hz")
        print(f" Duration: {raw.times[-1]:.1f} seconds")
        
        return raw
                        
    except Exception as e:
        print(f"Failed to load real data: {e}")
        return None




def apply_mne_pipeline(raw):
    """Apply MNE-based filtering pipeline."""
    after_notch = apply_notch_filter(raw, notch_freq=60.0)
    after_bandpass = apply_bandpass_filter(after_notch, l_freq=0.5, h_freq=40.0)
    after_avgref = apply_average_reference(after_bandpass)
    
    return raw, after_notch, after_bandpass, after_avgref


def apply_custom_pipeline(raw):
    """Apply custom SciPy/NumPy filtering pipeline."""
    raw_data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    after_notch = apply_custom_notch(raw_data, sfreq, notch_freq=60.0)
    after_bandpass = apply_custom_scipy_filter(after_notch, sfreq, 
                                              filter_type='bandpass', 
                                              frequencies=(0.5, 40.0))
    after_avgref = apply_custom_reference(after_bandpass, ref_channels='average')
    
    return raw_data, after_notch, after_bandpass, after_avgref


def plot_pipeline_comparison():
    """Create visual comparison plots."""
    # Try to load real data first
    raw = load_real_eeg_data()
    
    if raw is None:
        print('did not manage to load real data')
        exit()
    
    # Select channel and time window for plotting
    try:
        ch_idx = raw.ch_names.index(CHANNEL)
        print(f"Channel {CHANNEL} found at index {ch_idx}")
    except ValueError:
        print(f"Channel {CHANNEL} not found in data")
        print(f"Available channels: {raw.ch_names}")
        exit()
    ch_name = CHANNEL
    sfreq = raw.info['sfreq']
    
    # Use first 10 seconds
    time_window = int(TIME_WINDOW * sfreq)
    time_vector = np.arange(time_window) / sfreq
    
    # Apply both pipelines
    mne_stages = apply_mne_pipeline(raw)
    custom_stages = apply_custom_pipeline(raw)
    
    # Create comparison plots
    fig, axes = plt.subplots(4,1, figsize=(16, 8))
    
    # MNE Pipeline (top row)
    mne_labels = ['Raw', 'After Notch', 'After Bandpass', 'After Avg Ref']
    for i, (stage, label) in enumerate(zip(mne_stages, mne_labels)):
        if hasattr(stage, 'get_data'):  # MNE Raw object
            data = stage.get_data()[ch_idx, :time_window]
        else:
            data = stage[ch_idx, :time_window]
        
        axes[i].plot(time_vector, data, 'b-', linewidth=0.8)
    
    # Custom Pipeline (bottom row)
    custom_labels = ['Raw', 'After Notch', 'After Bandpass', 'After Avg Ref']
    for i, (stage, label) in enumerate(zip(custom_stages, custom_labels)):
        data = stage[ch_idx, :time_window]
        
        axes[i].plot(time_vector, data, 'r-', linewidth=0.8)
        axes[i].set_title(f'{label}', fontweight='bold')
        axes[i].set_ylabel('Amplitude (µV)', fontsize=10)
        axes[i].grid(True, alpha=0.3)
        axes[i].set_xlabel('Time (s)', fontsize=10)
    if i == 3:
        axes[i].legend(['MNE', 'Custom'], fontsize=10)
    
    fig.suptitle(f'EEG Filtering Pipeline Comparison - Channel: {ch_name}', 
                 fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    # Print comparison statistics
    if hasattr(mne_stages[-1], 'get_data'):
        mne_final = mne_stages[-1].get_data()[ch_idx, :time_window]
    else:
        mne_final = mne_stages[-1][ch_idx, :time_window]
    
    custom_final = custom_stages[-1][ch_idx, :time_window]
    
    print(f"Channel: {ch_name}")
    print(f"Sampling rate: {sfreq} Hz") 



if __name__ == "__main__":
    plot_pipeline_comparison()