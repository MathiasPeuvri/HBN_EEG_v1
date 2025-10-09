"""
Data Augmentation Strategies for EEG Contrastive Learning

Implements 6 augmentation techniques from Mohsenvand et al. (2020):
1. Amplitude scaling
2. Time shift
3. DC shift
4. Zero masking
5. Additive Gaussian noise
6. Band-stop filtering

All functions operate on numpy arrays of shape (n_channels, n_samples)
"""

import numpy as np
from scipy.signal import butter, filtfilt
from .config import transformation_ranges, FS


def amplitude_scaling(data: np.ndarray, scaling_factor: float) -> np.ndarray:
    """
    Scale amplitude of EEG signal by a multiplicative factor.
    """
    return data * scaling_factor


def time_shift_simple(data: np.ndarray, shift_percent: float) -> np.ndarray:
    """
    Apply circular time shift to EEG signal (simplified version).

    This is a SIMPLIFIED implementation using np.roll for circular shift.
    Creates artificial discontinuities at boundaries.

    TODO: Implement advanced version with access to continuous recording
    to avoid boundary discontinuities. See future task in TASK.md.

    Args:
        data: EEG signal (n_channels, n_samples)
        shift_percent: Shift as percentage of window length (e.g., 0.1 = 10%)

    Returns:
        Time-shifted signal of same shape
    """
    n_samples = data.shape[-1]
    shift_samples = int(shift_percent * n_samples)
    return np.roll(data, shift_samples, axis=-1)


def dc_shift(data: np.ndarray, shift_uV: float) -> np.ndarray:
    """
    Add constant DC offset to EEG signal.
    """
    return data + shift_uV


def zero_masking(data: np.ndarray, mask_percent: float) -> np.ndarray:
    """
    Randomly mask a contiguous portion of the signal by setting to zero.
    """
    masked_data = data.copy()
    n_samples = data.shape[-1]

    mask_length = int(mask_percent * n_samples)
    if mask_length > 0:
        start_idx_max = n_samples - mask_length
        start_idx = np.random.randint(0, start_idx_max + 1)
        # Apply zero mask to all channels at same time segment
        masked_data[:, start_idx:start_idx + mask_length] = 0

    return masked_data


def additive_gaussian_noise(data: np.ndarray, noise_scale: float) -> np.ndarray:
    """
    Add Gaussian noise scaled by signal standard deviation.
    """
    data_std = np.std(data)
    noise = np.random.normal(0, noise_scale * data_std, data.shape)
    return data + noise


def band_stop_filter(data: np.ndarray, center_freq_Hz: float, fs: int = FS) -> np.ndarray:
    """
    Apply 5Hz bandwidth band-stop (notch) filter to EEG signal.
    """
    lowcut = center_freq_Hz - 2.5
    highcut = center_freq_Hz + 2.5
    nyquist = fs / 2

    # Ensure filter frequencies are valid
    if lowcut <= 1.0 or highcut >= (nyquist - 1.0):
        return data

    low = lowcut / nyquist
    high = highcut / nyquist

    # Additional safety check
    if low <= 0.01 or high >= 0.99 or low >= high:
        return data

    try:
        b, a = butter(4, [low, high], btype='bandstop')
        # Apply filter to each channel
        filtered_data = np.array([filtfilt(b, a, channel) for channel in data])
        return filtered_data
    except (ValueError, np.linalg.LinAlgError):
        # If filter design fails, return original data
        return data


def apply_augmentations(
    data: np.ndarray,
    augmentation_ranges: dict = None,
    num_augmentations: int = 2
) -> tuple:
    """
    Apply random selection of augmentations to EEG signal.

    Args:
        data: EEG signal (n_channels, n_samples)
        augmentation_ranges: Dictionary of augmentation parameter ranges
        num_augmentations: Number of augmentations to apply

    Returns:
        augmented_data: Augmented signal (n_channels, n_samples)
        aug_info: List of (augmentation_name, parameter_value) tuples
    """
    if augmentation_ranges is None:
        augmentation_ranges = transformation_ranges

    # Available augmentations
    available_augs = ['amplitude_scale', 'time_shift', 'DC_shift', 'zero_masking', 'gaussian_noise', 'band_stop']

    # Randomly select augmentations (without replacement)
    num_augmentations = min(num_augmentations, len(available_augs))
    selected_augs = np.random.choice(available_augs, size=num_augmentations, replace=False)

    # Apply augmentations sequentially
    augmented_data = data.copy()
    aug_info = []

    for aug_name in selected_augs:
        if aug_name == 'amplitude_scale':
            param = np.random.uniform(*augmentation_ranges['amplitude_scale'])
            augmented_data = amplitude_scaling(augmented_data, param)
            aug_info.append((aug_name, f'{param:.2f}x'))

        elif aug_name == 'time_shift':
            param = np.random.uniform(*augmentation_ranges['time_shift_percent_twindow'])
            augmented_data = time_shift_simple(augmented_data, param)
            aug_info.append((aug_name, f'{param*100:.1f}%'))

        elif aug_name == 'DC_shift':
            param = np.random.uniform(*augmentation_ranges['DC_shift_µV'])
            augmented_data = dc_shift(augmented_data, param)
            aug_info.append((aug_name, f'{param:.2f}µV'))

        elif aug_name == 'zero_masking':
            param = np.random.uniform(*augmentation_ranges['zero_masking_percent_twindow'])
            augmented_data = zero_masking(augmented_data, param)
            aug_info.append((aug_name, f'{param*100:.1f}%'))

        elif aug_name == 'gaussian_noise':
            param = np.random.uniform(*augmentation_ranges['additive_gaussian_noise'])
            augmented_data = additive_gaussian_noise(augmented_data, param)
            aug_info.append((aug_name, f'σ={param:.2f}'))

        elif aug_name == 'band_stop':
            param = np.random.uniform(*augmentation_ranges['band5hz_stop_Hzstart'])
            augmented_data = band_stop_filter(augmented_data, param)
            aug_info.append((aug_name, f'{param:.1f}Hz'))

    return augmented_data, aug_info


def create_augmented_pair(
    data: np.ndarray,
    augmentation_ranges: dict = None,
    num_augmentations: int = 2
) -> tuple:
    """
    Create two differently augmented versions of the same EEG sample (positive pair).
    """
    view1, info1 = apply_augmentations(data, augmentation_ranges, num_augmentations)
    view2, info2 = apply_augmentations(data, augmentation_ranges, num_augmentations)

    return view1, view2, info1, info2
