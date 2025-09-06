"""
EEG filtering functions for preprocessing pipeline.

Hybrid approach: Use MNE for complex filtering (proper FIR design, edge handling),
SciPy for custom filters when needed.
"""

import mne
import numpy as np
from scipy import signal
from scipy.signal import iirnotch, filtfilt
from typing import Optional, Union, Tuple, Literal

def apply_notch_filter(raw: mne.io.Raw, notch_freq: float = 60.0) -> mne.io.Raw:
    """
    Apply notch filter to remove power line interference and harmonics.
    Args:
        notch_freq: Frequency to notch (default 60Hz for US power line)
        
    Returns:
        Filtered Raw object
    """
    # Apply notch filter at fundamental frequency and harmonics
    # Only include harmonics that are below Nyquist frequency
    nyquist_freq = raw.info['sfreq'] / 2
    harmonics = []
    for harmonic in [notch_freq, notch_freq*2, notch_freq*3]:
        if harmonic < nyquist_freq:
            harmonics.append(harmonic)
    
    raw_filtered = raw.copy()
    if harmonics:  # Only apply filter if we have valid harmonics
        raw_filtered.notch_filter(harmonics, verbose=False)
    return raw_filtered


def apply_bandpass_filter(raw: mne.io.Raw, l_freq: float = 0.5, h_freq: float = 40.0) -> mne.io.Raw:
    """
    Apply bandpass filter to EEG data using MNE.
    Args:
        l_freq: Low-frequency cutoff (Hz)
        h_freq: High-frequency cutoff (Hz)
        
    Returns:
        Filtered Raw object
    """
    raw_filtered = raw.copy()
    raw_filtered.filter(l_freq=l_freq, h_freq=h_freq, method='fir', 
                       phase='zero', fir_design='firwin', verbose=False)
    return raw_filtered


def apply_custom_scipy_filter(data: np.ndarray, sfreq: float, 
                             filter_type: str = 'bandpass',
                             frequencies: Union[float, Tuple[float, float]] = (0.5, 40.0)) -> np.ndarray:
    """
    Apply custom filter using SciPy 
    Args:
        data: EEG data array (channels x time)
        sfreq: Sampling frequency
        filter_type: 'bandpass', 'lowpass', 'highpass', 'bandstop'
        frequencies: Cutoff frequency(ies)
        
    Returns:
        Filtered data array
    """
    if filter_type == 'bandpass' and len(frequencies) == 2:
        bandpassfreq = list(frequencies)
        if bandpassfreq[0] == 0:
            passfilt = 'lowpass'
            bandpassfreq = bandpassfreq[1]
        elif bandpassfreq[1] > 200:
            passfilt = 'highpass'
            bandpassfreq = bandpassfreq[0]
        else:
            passfilt = 'bandpass'
    else:
        passfilt = filter_type
        bandpassfreq = frequencies
    
    sos = signal.butter(3, bandpassfreq, passfilt, fs=sfreq, output='sos')
    filtered_data = signal.sosfilt(sos, data, axis=1)
    return filtered_data


def apply_average_reference(raw: mne.io.Raw) -> mne.io.Raw:
    """
    Apply average reference to EEG with MNE
    Args:
        raw: MNE Raw object
        
    Returns:
        Re-referenced Raw object
    """
    raw_filtered = raw.copy()
    raw_filtered.set_eeg_reference('average', projection=True, verbose=False)
    raw_filtered.apply_proj(verbose=False)
    return raw_filtered


def apply_custom_reference(data: np.ndarray, ref_channels: Union[list, str] = 'average') -> np.ndarray:
    """
    Apply custom referencing - Custom  schemes 
    Args:
        data: EEG data array (channels x time)
        ref_channels: 'average', 'median', or list of channel indices
        
    Returns:
        Re-referenced data array
    """
    if ref_channels == 'average':
        reference_signal = np.mean(data, axis=0, keepdims=True)
        referenced_data = data - reference_signal
    elif ref_channels == 'median':
        reference_signal = np.median(data, axis=0, keepdims=True)
        referenced_data = data - reference_signal
    elif isinstance(ref_channels, list):
        reference_signal = np.mean(data[ref_channels], axis=0, keepdims=True)
        referenced_data = data - reference_signal
    else:
        referenced_data = data.copy()
    
    return referenced_data


def apply_custom_notch(data: np.ndarray, sfreq: float, notch_freq: float = 60.0) -> np.ndarray:
    """
    Apply custom notch filter using SciPy's IIR notch filter.
    Based on IEDetector implementation using iirnotch and filtfilt.
    
    Args:
        data: EEG data array (channels x time)
        sfreq: Sampling frequency
        notch_freq: Frequency to notch (default 60Hz)
        
    Returns:
        Filtered data array
    """
    # Apply notch filter for fundamental frequency and harmonics
    filtered_data = data.copy()
    for freq in [notch_freq, notch_freq*2, notch_freq*3]:  # 60, 120, 180 Hz
        if freq < sfreq/2:  # Ensure frequency is below Nyquist
            b, a = iirnotch(freq / (sfreq / 2), 30)  # Quality factor of 30
            filtered_data = filtfilt(b, a, filtered_data, axis=1)
    
    return filtered_data


def zscore_normZ2(signals, axis = None):
    """
    previous 
    """
    if axis is None or axis == 1:
        mean = np.mean(signals)
        std = np.std(signals)
    else:
        mean = np.mean(signals, axis=axis)
        std = np.std(signals, axis=axis)

    signals = (signals - mean) / std
    return signals


def zscore(data: np.ndarray, method: Literal['channel_wise', 'globalnorm', 'spatial'] = 'channel_wise') -> np.ndarray:
    """
    Z-score normalization for EEG data.
    !!! Assumes data shape is (channels x time)
    
    Methods:
    - channel_wise: Each channel by its own temporal mean/std (axis=1)
    - globalnorm: All data by single global mean/std 
    - spatial: Each time point by spatial mean/std across channels (axis=0)
    """
    if len(data.shape) != 2:
        raise ValueError(f"Expected 2D data (channels x time), got shape {data.shape}")
        
    if method == 'channel_wise':
        mean = np.mean(data, axis=1, keepdims=True)  # (n_channels x 1)
        std = np.std(data, axis=1, keepdims=True)
    elif method == 'globalnorm':
        mean = np.mean(data)  # scalar
        std = np.std(data)
    elif method == 'spatial':
        mean = np.mean(data, axis=0, keepdims=True)  # (1 x n_times)
        std = np.std(data, axis=0, keepdims=True)
    
    if isinstance(std, np.ndarray):
        std[std == 0] = 1
    elif std == 0: 
        std = 1

    return (data - mean) / std


def preprocess_data(data, notch_freq: float = 60.0, bandpass_freq: Tuple[float, float] = (0.5, 40.0), 
                    ref_channels: str = 'average', 
                    zscore_method: Literal['channel_wise', 'globalnorm', 'spatial'] = 'channel_wise'):
    """ data from SimpleHBNLoader.get_data() """

    raw = data['raw']
    raw_data = raw.get_data()
    sfreq = raw.info['sfreq']

    after_notch = apply_custom_notch(raw_data, sfreq, notch_freq=notch_freq)
    after_bandpass = apply_custom_scipy_filter(after_notch, sfreq, 
                                                filter_type='bandpass', 
                                                frequencies=bandpass_freq)
    after_avgref = apply_custom_reference(after_bandpass, ref_channels=ref_channels)
    zscore_data = zscore(after_avgref, method=zscore_method)

    preprocessed_data = zscore_data
    return preprocessed_data





def detect_bad_channels(raw: mne.io.Raw, threshold: float = 3.0) -> list:
    """
    Placeholder for bad channel detection
        
    Returns:
        List of bad channel names
    """
    # TODO: Implement MNE bad channel detection
    # Options:
    # 1. mne.preprocessing.find_bad_channels_zscore(raw, threshold=threshold)
    # 2. mne.preprocessing.find_bad_channels_maxwell(raw) - if you have fine calibration
    # 3. Custom: compute channel variances and flag outliers
    
    print(f"TODO: Detecting bad channels with MNE (threshold: {threshold})")
    return []

