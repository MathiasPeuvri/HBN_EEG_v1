#!/usr/bin/env python3
"""
Shared utilities for task visualizations.

Contains common functions, constants, and configurations used across
all task visualization scripts.
"""

import sys
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path

# Add the project root directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.loader import SimpleHBNLoader
from src.loader.config import DatabaseLoaderConfig

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Data configuration
DATASET_ROOT = Path(__file__).parent.parent / 'database'
DATASET_NAME = 'R1_L100'
DEFAULT_SUBJECT = 'NDARAC904DMU'
DEFAULT_RUN = 2

# Visualization settings
FIGURE_DPI = 300
EEG_CHANNELS_TO_DISPLAY = 2
CHANNEL_SEPARATION_FACTOR = 3

# Task display limits
DEFAULT_TRIAL_COUNT = None
DEFAULT_STIM_COUNT = None

# Resting state thresholds
MIN_SEGMENT_DURATION = 1
MIN_LABEL_DURATION = 8

# Color schemes
RESTING_STATE_COLORS = {'open': '#FFB6C1', 'closed': '#87CEEB'}
INSTRUCTION_COLORS = {'open': 'red', 'close': 'blue'}
CONTRAST_COLORS = {'left': '#FFA500', 'right': '#6495ED'}
MOVIE_COLORS = {
    'DespicableMe': '#FFD700',      # Gold
    'DiaryOfAWimpyKid': '#32CD32',  # Lime Green
    'FunwithFractals': '#9370DB',   # Medium Purple
    'ThePresent': '#FF6347'         # Tomato
}
SEQUENCE_COLORS = {
    'dot_on': '#FF6B6B',           # Light red for active dots
    'block': '#4ECDC4',            # Teal for block boundaries
    'correct': '#32CD32',          # Lime green for correct
    'incorrect': '#DC143C',        # Crimson for incorrect
    'response_period': '#FFE4B5'   # Moccasin for response period
}
SYMBOL_COLORS = {
    'correct': '#32CD32',          # Lime green for correct responses
    'incorrect': '#DC143C',        # Crimson for incorrect responses
    'page_boundary': '#808080',    # Gray for page boundaries
    'trial_bar': '#87CEEB'         # Sky blue for trial background
}

# =============================================================================
# SHARED UTILITIES
# =============================================================================

def create_loader(dataset_root=None, dataset_name=None):
    """Create and return configured data loader.
    
    Args:
        dataset_root: Optional path to dataset root (defaults to DATASET_ROOT)
        dataset_name: Optional dataset name (defaults to DATASET_NAME)
    """
    root = dataset_root if dataset_root is not None else DATASET_ROOT
    name = dataset_name if dataset_name is not None else DATASET_NAME
    config = DatabaseLoaderConfig(data_root=root, dataset_name=name)
    return SimpleHBNLoader(config)

def load_task_data(loader, subject_id, task_name, run=None):
    """Load EEG data and events for a task with proper error handling."""
    data = loader.get_data(subject_id, task_name, run=run, include_events=True)
    raw = data['raw']
    events_df = data['events']
    
    # Extract real sampling rate and duration from MNE object
    sampling_rate = raw.info['sfreq']
    duration = raw.times[-1]  # Actual EEG duration
    
    print(f"Loaded {task_name}: {duration:.1f}s at {sampling_rate}Hz")
    return raw, events_df, sampling_rate, duration

def add_eeg_subplot(ax, raw, events_df, max_time, event_markers=None):
    """Add EEG signals to subplot with proper time alignment."""
    eeg_data = raw.get_data()
    sampling_rate = raw.info['sfreq']
    
    if eeg_data.shape[0] < EEG_CHANNELS_TO_DISPLAY:
        raise ValueError(f"Need at least {EEG_CHANNELS_TO_DISPLAY} channels, got {eeg_data.shape[0]}")
    
    # Use actual EEG duration, limited by max_time if needed
    actual_duration = min(raw.times[-1], max_time)
    end_sample = int(actual_duration * sampling_rate)
    eeg_subset = eeg_data[:EEG_CHANNELS_TO_DISPLAY, :end_sample]
    time_axis = raw.times[:end_sample]
    
    # Plot channels with visual separation
    channel_offset = np.std(eeg_subset) * CHANNEL_SEPARATION_FACTOR
    ax.plot(time_axis, eeg_subset[0, :], 'b-', linewidth=0.8, label='Channel 1 (e.g., Fp1)')
    ax.plot(time_axis, eeg_subset[1, :] + channel_offset, 'r-', linewidth=0.8, label='Channel 2 (e.g., O1)')
    
    # Add event markers if provided
    if event_markers is not None and not event_markers.empty:
        for _, event in event_markers.iterrows():
            if event['onset'] <= actual_duration:
                color = INSTRUCTION_COLORS.get(event.get('type', 'open'), 'black')
                ax.axvline(event['onset'], color=color, linestyle='--', alpha=0.7, linewidth=1.5)
    
    # Format subplot
    ax.set_xlim(0, max_time)
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('EEG Amplitude (μV)', fontsize=10)
    ax.set_title('EEG Signals (2 channels)', fontsize=12)
    ax.legend(fontsize=9)