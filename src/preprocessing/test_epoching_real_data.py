#!/usr/bin/env python3
"""
Visual test for epoch segmentation using real HBN data.
Tests the segment_continuous_numpy function with different overlap settings.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, '/home/mts/HBN_EEG_Analysis')

# Import our epoching function
from src.preprocessing.epoching import segment_continuous_numpy
from src.loader import SimpleHBNLoader, SimpleConfig

SUBJECT = "NDARAC904DMU"
TASK = "RestingState"
CHANNEL = "E45"  # channels E1 to E128 (+ Cz)
TIME_WINDOW = 10  # seconds to visualize
EPOCH_LENGTH = 2.0  # seconds per epoch

def load_real_eeg_data():
    """
    Load real EEG data using the existing loader.
    
    Returns:
        mne.io.Raw: Real EEG data or None if failed
    """
    try:
        config = SimpleConfig(data_root=Path("/home/mts/EFG2025_HBN-EEG_databse/data"))
        loader = SimpleHBNLoader(config)
        
        data = loader.get_data(SUBJECT, TASK)
        raw = data['raw']
        
        print(f"✓ Loaded real EEG data: Subject {SUBJECT}, Task {TASK}")
        print(f"  Data shape: {raw.get_data().shape}")
        print(f"  Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  Duration: {raw.times[-1]:.1f} seconds")
        print()
        
        return raw
        
    except Exception as e:
        print(f"Failed to load real data: {e}")
        return None


def test_epoching_overlaps():
    """Test epoching with 0% and 50% overlap and visualize results."""
    
    # Load real data
    raw = load_real_eeg_data()
    if raw is None:
        print("Could not load real data, exiting.")
        return
    
    # Get channel index
    try:
        ch_idx = raw.ch_names.index(CHANNEL)
        print(f"✓ Channel {CHANNEL} found at index {ch_idx}")
    except ValueError:
        print(f"Channel {CHANNEL} not found in data")
        print(f"Available channels: {raw.ch_names[:10]}...")  # Show first 10
        return
    
    # Get data and parameters
    raw_data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    # Use first 20 seconds for testing (gives us more epochs to visualize)
    test_duration = 20  # seconds
    test_samples = int(test_duration * sfreq)
    test_data = raw_data[:, :test_samples]
    
    print(f"Testing with {test_duration}s of data ({test_samples} samples)")
    print(f"Epoch length: {EPOCH_LENGTH}s")
    print()
    
    # Test 0% overlap
    print("=== Testing 0% overlap ===")
    epochs_0 = segment_continuous_numpy(test_data, sfreq, EPOCH_LENGTH, overlap=0.0)
    print(f'Epochs shape: {epochs_0.shape}')
    
    # Test 50% overlap  
    print("=== Testing 50% overlap ===")
    epochs_50 = segment_continuous_numpy(test_data, sfreq, EPOCH_LENGTH, overlap=0.5)
    print(f'Epochs shape: {epochs_50.shape}')
    
    # Create visualization
    fig, axes = plt.subplots(3, 1, figsize=(15, 10))
    
    # Time vectors
    time_full = np.arange(test_samples) / sfreq
    time_epoch = np.arange(int(EPOCH_LENGTH * sfreq)) / sfreq
    
    # Plot 1: Original continuous signal (first 10 seconds for clarity)
    display_samples = int(TIME_WINDOW * sfreq)
    axes[0].plot(time_full[:display_samples], test_data[ch_idx, :display_samples], 'k-', linewidth=0.8)
    axes[0].set_title(f'Original Continuous Signal - Channel {CHANNEL} (First {TIME_WINDOW}s)', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude (µV)')
    axes[0].grid(True, alpha=0.3)
    
    # Add epoch boundaries for 0% overlap
    epoch_samples = int(EPOCH_LENGTH * sfreq)
    for i in range(int(TIME_WINDOW // EPOCH_LENGTH) + 1):
        boundary_time = i * EPOCH_LENGTH
        if boundary_time <= TIME_WINDOW:
            axes[0].axvline(boundary_time, color='red', linestyle='--', alpha=0.7, linewidth=1)
    
    # Plot 2: Epochs with 0% overlap
    n_epochs_to_show = min(epochs_0.shape[0], int(TIME_WINDOW // EPOCH_LENGTH))
    colors_0 = plt.cm.Set1(np.linspace(0, 1, n_epochs_to_show))
    
    for i in range(n_epochs_to_show):
        axes[1].plot(time_epoch + i * EPOCH_LENGTH, epochs_0[i, ch_idx, :], 
                    color=colors_0[i], linewidth=1.2, label=f'Epoch {i+1}')
    
    axes[1].set_title(f'Segmented Epochs - 0% Overlap ({epochs_0.shape[0]} total epochs)', 
                     fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Amplitude (µV)')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Plot 3: Epochs with 50% overlap
    n_epochs_to_show_50 = min(epochs_50.shape[0], int(TIME_WINDOW // (EPOCH_LENGTH * 0.5)))
    colors_50 = plt.cm.Set2(np.linspace(0, 1, n_epochs_to_show_50))
    
    step_time_50 = EPOCH_LENGTH * 0.5  # 50% overlap means 50% step
    
    for i in range(n_epochs_to_show_50):
        axes[2].plot(time_epoch + i * step_time_50, epochs_50[i, ch_idx, :], 
                    color=colors_50[i], linewidth=1.2, alpha=0.8, label=f'Epoch {i+1}')
    
    axes[2].set_title(f'Segmented Epochs - 50% Overlap ({epochs_50.shape[0]} total epochs)', 
                     fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Amplitude (µV)')
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print("=== Summary Statistics ===")
    print(f"Original data duration: {test_duration}s ({test_samples} samples)")
    print(f"Epoch length: {EPOCH_LENGTH}s ({int(EPOCH_LENGTH * sfreq)} samples)")
    print(f"Channel: {CHANNEL}")
    print()
    print(f"0% overlap: {epochs_0.shape[0]} epochs")
    print(f"  Coverage: {epochs_0.shape[0] * EPOCH_LENGTH:.1f}s "
          f"({epochs_0.shape[0] * EPOCH_LENGTH / test_duration * 100:.1f}% of data)")
    print()
    print(f"50% overlap: {epochs_50.shape[0]} epochs") 
    print(f"  Coverage: {(epochs_50.shape[0] - 1) * EPOCH_LENGTH * 0.5 + EPOCH_LENGTH:.1f}s "
          f"({((epochs_50.shape[0] - 1) * EPOCH_LENGTH * 0.5 + EPOCH_LENGTH) / test_duration * 100:.1f}% of data)")
    print()
    
    # Data quality checks
    print("=== Data Quality Checks ===")
    print(f"0% overlap epochs shape: {epochs_0.shape}")
    print(f"50% overlap epochs shape: {epochs_50.shape}")
    
    # Check for any NaN or infinite values
    if np.any(np.isnan(epochs_0)) or np.any(np.isinf(epochs_0)):
        print("⚠️  Warning: Found NaN or infinite values in 0% overlap epochs")
    else:
        print("✓ 0% overlap epochs: No NaN or infinite values")
        
    if np.any(np.isnan(epochs_50)) or np.any(np.isinf(epochs_50)):
        print("⚠️  Warning: Found NaN or infinite values in 50% overlap epochs")
    else:
        print("✓ 50% overlap epochs: No NaN or infinite values")


if __name__ == "__main__":
    test_epoching_overlaps()