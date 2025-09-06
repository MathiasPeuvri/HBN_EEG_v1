#!/usr/bin/env python3
"""
Visual test for event-based epoch segmentation using real HBN data.
Tests both MNE and NumPy implementations with proper event type separation.
"""

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/mts/HBN_EEG_Analysis')

# Import our epoching functions
from src.preprocessing.epoching import segment_by_events_mne, segment_by_events_numpy
from src.loader import SimpleHBNLoader, SimpleConfig

# Configuration
SUBJECT = "NDARAC904DMU"
TASK = "contrastChangeDetection"
RUN = 1
CHANNEL = "E45"
TIME_WINDOW = 120  # seconds of continuous data to show
TMIN, TMAX = -2.0, 1.0  # Event epoch window
TARGET_EVENTS = ['left_target', 'right_target']


def load_task_data():
    """
    Load task EEG data and events.
    
    Returns:
        tuple: (raw, events_df) or (None, None) if failed
    """
    try:
        config = SimpleConfig(data_root=Path("/home/mts/EFG2025_HBN-EEG_databse/data"))
        loader = SimpleHBNLoader(config)
        
        print(f"Loading {TASK} task for subject {SUBJECT} (run {RUN})")
        data = loader.get_data(SUBJECT, TASK, run=RUN, include_events=True)
        
        raw = data['raw']
        events_df = data['events']
        
        print(f"  Shape: {raw.get_data().shape}, Sampling rate: {raw.info['sfreq']} Hz")
        print(f"  Duration: {raw.times[-1]:.1f} seconds, Events: {len(events_df)} total events")
        print()
        
        return raw, events_df
        
    except Exception as e:
        print(f"Failed to load data: {e}")
        return None, None


def process_events_by_type(raw, events_df, sfreq):
    """
    Process a unique type of event.
    
    Args:
        raw: MNE Raw object
        events_df: Events DataFrame
        sfreq: Sampling frequency
        
    Returns:
        tuple: (mne_epochs_dict, numpy_epochs_dict, event_info)
    """
    print("=== Event Processing ===")
    print(f"Event types found: {events_df['value'].unique()}")
    
    # Event ID mapping
    event_ids = {'left_target': 8, 'right_target': 9}
    
    mne_epochs_dict = {}
    numpy_epochs_dict = {}
    event_info = {}
    
    # Process each event type separately
    for event_type in TARGET_EVENTS:
        print(f"\n--- Processing {event_type} ---")
        
        # Filter events for this type only
        events_this_type = events_df[events_df['value'] == event_type].copy()
        print(f"Found {len(events_this_type)} {event_type} events")
        
        if len(events_this_type) == 0:
            print(f"No {event_type} events found, skipping")
            continue
        
        # Create MNE events array for this type
        mne_events = np.zeros((len(events_this_type), 3), dtype=int)
        mne_events[:, 0] = (events_this_type['onset'] * sfreq).astype(int)
        mne_events[:, 1] = 0  # Duration
        mne_events[:, 2] = event_ids[event_type]  # Event ID
        
        # Create MNE epochs for this event type
        try:
            event_id_single = {event_type: event_ids[event_type]}
            mne_epochs = segment_by_events_mne(
                raw, mne_events, event_id_single, tmin=TMIN, tmax=TMAX
            )
            mne_epochs_dict[event_type] = mne_epochs
            print(f"MNE epochs shape: {mne_epochs.get_data().shape}")
        except Exception as e:
            print(f"MNE epoching failed for {event_type}: {e}")
            continue
        
        # Create NumPy epochs for this event type
        try:
            raw_data = raw.get_data()
            event_times_samples = (events_this_type['onset'].values * sfreq).astype(int)
            numpy_epochs = segment_by_events_numpy(
                raw_data, sfreq, event_times_samples, tmin=TMIN, tmax=TMAX
            )
            numpy_epochs_dict[event_type] = numpy_epochs
            print(f"NumPy epochs shape: {numpy_epochs.shape}")
        except Exception as e:
            print(f"NumPy epoching failed for {event_type}: {e}")
            continue
        
        # Store event information
        event_info[event_type] = {
            'count': len(events_this_type),
            'event_times': events_this_type['onset'].values,
            'color': 'blue' if event_type == 'left_target' else 'red'
        }
    
    return mne_epochs_dict, numpy_epochs_dict, event_info


def create_visualization(raw, mne_epochs_dict, numpy_epochs_dict, event_info, ch_idx, sfreq):
    """
    Create clean visualization of event-based epoching.
    """
    fig, axes = plt.subplots(4, 1, figsize=(16, 12))
    
    # Time axis for epochs
    epoch_samples = int((TMAX - TMIN) * sfreq) + 1
    time_epoch = np.linspace(TMIN, TMAX, epoch_samples)
    
    # 1. Continuous signal with event markers
    display_samples = int(TIME_WINDOW * sfreq)
    time_continuous = np.arange(display_samples) / sfreq
    raw_data = raw.get_data()
    
    axes[0].plot(time_continuous, raw_data[ch_idx, :display_samples], 'k-', linewidth=0.8)
    axes[0].set_title(f'Continuous EEG Signal with Event Markers - Channel {CHANNEL}', 
                     fontsize=14, fontweight='bold')
    axes[0].set_ylabel('Amplitude (µV)')
    
    # Add event markers
    legend_labels = ['EEG Signal']
    for event_type in TARGET_EVENTS:
        if event_type in event_info:
            color = event_info[event_type]['color']
            event_times = event_info[event_type]['event_times']
            for event_time in event_times:
                if event_time < TIME_WINDOW:
                    axes[0].axvline(event_time, color=color, linestyle='--', alpha=0.7, linewidth=1)
            legend_labels.append(f'{event_type.replace("_", " ").title()}')
    
    axes[0].legend(legend_labels, loc='upper right')
    axes[0].grid(True, alpha=0.3)
    
    # 2. MNE epochs - sample from each type
    n_show = 3  # Show 3 epochs per type
    
    for event_type in TARGET_EVENTS:
        if event_type in mne_epochs_dict:
            mne_epochs = mne_epochs_dict[event_type]
            mne_data = mne_epochs.get_data()
            color = event_info[event_type]['color']
            
            for i in range(min(n_show, len(mne_data))):
                axes[1].plot(time_epoch, mne_data[i, ch_idx, :], 
                           color=color, alpha=0.7, linewidth=1,
                           label=f'{event_type}' if i == 0 else "")
    
    axes[1].axvline(0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    axes[1].set_title(f'MNE Event Epochs (first {n_show} per type)', 
                     fontsize=14, fontweight='bold')
    axes[1].set_ylabel('Amplitude (µV)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # 3. NumPy epochs - sample from each type
    for event_type in TARGET_EVENTS:
        if event_type in numpy_epochs_dict:
            numpy_epochs = numpy_epochs_dict[event_type]
            color = event_info[event_type]['color']
            
            for i in range(min(n_show, len(numpy_epochs))):
                axes[2].plot(time_epoch, numpy_epochs[i, ch_idx, :], 
                           color=color, alpha=0.7, linewidth=1,
                           label=f'{event_type}' if i == 0 else "")
    
    axes[2].axvline(0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    axes[2].set_title(f'NumPy Event Epochs (first {n_show} per type)', 
                     fontsize=14, fontweight='bold')
    axes[2].set_ylabel('Amplitude (µV)')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    # 4. Average ERPs - MNE vs NumPy comparison
    for event_type in TARGET_EVENTS:
        if event_type in mne_epochs_dict and event_type in numpy_epochs_dict:
            # MNE average
            mne_data = mne_epochs_dict[event_type].get_data()
            mne_avg = np.mean(mne_data[:, ch_idx, :], axis=0)
            
            # NumPy average
            numpy_data = numpy_epochs_dict[event_type]
            numpy_avg = np.mean(numpy_data[:, ch_idx, :], axis=0)
            
            color = event_info[event_type]['color']
            axes[3].plot(time_epoch, mne_avg, color=color, linewidth=2, 
                        linestyle='-', label=f'{event_type} MNE')
            axes[3].plot(time_epoch, numpy_avg, color=color, linewidth=2, 
                        linestyle='--', label=f'{event_type} NumPy')
    
    axes[3].axvline(0, color='black', linestyle='-', alpha=0.8, linewidth=2)
    axes[3].set_title('Average Event-Related Potentials - MNE vs NumPy', 
                     fontsize=14, fontweight='bold')
    axes[3].set_xlabel('Time (s)')
    axes[3].set_ylabel('Amplitude (µV)')
    axes[3].legend()
    axes[3].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plt.savefig('/home/mts/HBN_EEG_Analysis/event_epoching_test.png', 
                dpi=150, bbox_inches='tight')
    print("Visualization saved as 'event_epoching_test.png'")
    plt.close()


def print_summary(mne_epochs_dict, numpy_epochs_dict, event_info):
    """Print summary statistics."""
    print("\n=== Summary Statistics ===")
    print(f"Channel analyzed: {CHANNEL}")
    print(f"Epoch window: {TMIN} to {TMAX} seconds")
    print()
    
    for event_type in TARGET_EVENTS:
        if event_type in event_info:
            print(f"{event_type}:")
            print(f"  Events found: {event_info[event_type]['count']}")
            if event_type in mne_epochs_dict:
                print(f"  MNE epochs: {len(mne_epochs_dict[event_type])}")
            if event_type in numpy_epochs_dict:
                print(f"  NumPy epochs: {len(numpy_epochs_dict[event_type])}")
            print()


def test_event_epoching():
    """Test event-based epoching with proper separation of event types."""
    
    # Load data
    raw, events_df = load_task_data()
    if raw is None:
        return
    
    # Get channel index
    try:
        ch_idx = raw.ch_names.index(CHANNEL)
        print(f"Channel {CHANNEL} found at index {ch_idx}")
    except ValueError:
        print(f"Channel {CHANNEL} not found")
        print(f"Available channels: {raw.ch_names[:10]}...")
        return
    
    sfreq = raw.info['sfreq']
    
    # Process events by type
    mne_epochs_dict, numpy_epochs_dict, event_info = process_events_by_type(raw, events_df, sfreq)
    
    # Create visualization
    create_visualization(raw, mne_epochs_dict, numpy_epochs_dict, event_info, ch_idx, sfreq)
    
    # Print summary
    print_summary(mne_epochs_dict, numpy_epochs_dict, event_info)


if __name__ == "__main__":
    test_event_epoching()