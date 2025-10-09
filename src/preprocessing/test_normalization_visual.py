#!/usr/bin/env python3
"""Visual test of EEG normalization strategies with real HBN data."""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/home/mts/HBN_EEG_v1')
from src.preprocessing.filters import zscore
from src.loader import SimpleHBNLoader, SimpleConfig

SUBJECT = "NDARAC904DMU"
TASK = "RestingState"
CHANNELS = ["E45", "E70", "E100"]  # frontal, parietal, occipital
TIME_SEC = 3

def load_data():
    """Load real EEG data."""
    try:
        config = SimpleConfig(data_root=Path("/home/mts/EFG2025_HBN-EEG_databse/data"))
        loader = SimpleHBNLoader(config)
        raw = loader.get_data(SUBJECT, TASK)['raw']
        print(f"Loaded: {raw.get_data().shape}, {raw.info['sfreq']} Hz")
        return raw
    except Exception as e:
        print(f"Failed: {e}")
        return None

def plot_comparison():
    """Compare normalization methods visually."""
    raw = load_data()
    if raw is None:
        return
    
    # Get subset
    data = raw.get_data()
    sfreq = raw.info['sfreq']
    samples = int(TIME_SEC * sfreq)
    time_vec = np.arange(samples) / sfreq
    plot_data = data[:, :samples]
    
    # Get channel indices
    ch_indices = []
    for ch in CHANNELS:
        try:
            ch_indices.append((ch, raw.ch_names.index(ch)))
        except ValueError:
            print(f"Channel {ch} not found")
    
    # Verify data format assumption
    print(f"Data shape: {plot_data.shape} (assuming channels x time)")
    
    # Apply normalizations
    methods = ['channel_wise', 'global', 'spatial']
    normalized = {m: zscore(plot_data, method=m) for m in methods}
    
    # Plot
    fig, axes = plt.subplots(len(methods), len(ch_indices), figsize=(12, 8))
    
    for row, method in enumerate(methods):
        for col, (ch_name, ch_idx) in enumerate(ch_indices):
            ax = axes[row, col]
            
            # Raw vs normalized
            ax.plot(time_vec, plot_data[ch_idx, :], 'k-', alpha=0.7, label='Raw')
            ax.plot(time_vec, normalized[method][ch_idx, :], 'r-', label=method.title())
            
            if row == 0:
                ax.set_title(ch_name)
            if col == 0:
                ax.set_ylabel(method.replace('_', ' ').title())
            if row == len(methods) - 1:
                ax.set_xlabel('Time (s)')
            
            ax.grid(alpha=0.3)
            if row == 0 and col == 0:
                ax.legend()
    
    plt.tight_layout()
    plt.show()
    
    # Print stats
    print(f"\nNormalization Effects on Channel {CHANNELS[0]}:")
    for method in methods:
        norm_signal = normalized[method][ch_indices[0][1], :]
        print(f"{method:12}: mean={np.mean(norm_signal):.3f}, std={np.std(norm_signal):.3f}")

if __name__ == "__main__":
    plot_comparison()