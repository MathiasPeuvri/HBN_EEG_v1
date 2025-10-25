"""
GPU-Accelerated Data Augmentation for EEG Contrastive Learning

PyTorch implementation of 6 augmentation techniques from Mohsenvand et al. (2020):
1. Amplitude scaling
2. Time shift
3. DC shift
4. Zero masking
5. Additive Gaussian noise
6. Band-stop filtering (using torchaudio)

All functions operate on torch.Tensor of shape (n_channels, n_samples) on GPU.
"""

import torch
import torchaudio.functional as F
from .config import transformation_ranges, FS


def amplitude_scaling_torch(data: torch.Tensor, scaling_factor: float) -> torch.Tensor:
    """
    Scale amplitude of EEG signal by a multiplicative factor.
    """
    return data * scaling_factor


def time_shift_torch(data: torch.Tensor, shift_percent: float) -> torch.Tensor:
    """
    Apply circular time shift to EEG signal.
    This is a SIMPLIFIED implementation using torch.roll for circular shift.
    Creates artificial discontinuities at boundaries.
    TODO: Implement advanced version with access to continuous recording
    to avoid boundary discontinuities.
    """
    n_samples = data.shape[-1]
    shift_samples = int(shift_percent * n_samples)
    return torch.roll(data, shift_samples, dims=-1)


def dc_shift_torch(data: torch.Tensor, shift_uV: float) -> torch.Tensor:
    """
    Add constant DC offset to EEG signal.
    """
    return data + shift_uV


def zero_masking_torch(data: torch.Tensor, mask_percent: float) -> torch.Tensor:
    """
    Randomly mask a contiguous portion of the signal by setting to zero.
    """
    masked_data = data.clone()
    n_samples = data.shape[-1]

    mask_length = int(mask_percent * n_samples)
    if mask_length > 0:
        start_idx_max = n_samples - mask_length
        start_idx = torch.randint(0, start_idx_max + 1, (1,), device=data.device).item()
        # Apply zero mask to all channels at same time segment
        masked_data[:, start_idx:start_idx + mask_length] = 0

    return masked_data


def additive_gaussian_noise_torch(data: torch.Tensor, noise_scale: float) -> torch.Tensor:
    """
    Add Gaussian noise scaled by signal standard deviation.
    """
    data_std = torch.std(data)
    noise = torch.randn_like(data) * (noise_scale * data_std)
    return data + noise


def band_stop_filter_torch(data: torch.Tensor, center_freq_Hz: float, fs: int = FS) -> torch.Tensor:
    """
    Apply 5Hz bandwidth band-stop (notch) filter to EEG signal using torchaudio.
    """
    lowcut = center_freq_Hz - 2.5
    highcut = center_freq_Hz + 2.5
    nyquist = fs / 2

    # Ensure filter frequencies are valid
    if lowcut <= 1.0 or highcut >= (nyquist - 1.0):
        return data

    # Additional safety check
    if lowcut <= 0.01 * nyquist or highcut >= 0.99 * nyquist or lowcut >= highcut:
        return data

    try:
        # Ensure data is 2D (channels, samples)
        original_shape = data.shape
        if data.dim() == 1:
            data = data.unsqueeze(0)

        # Use torchaudio's bandreject_biquad for band-stop filtering
        # Q factor controls the bandwidth: Q = center_freq / bandwidth
        # For 5Hz bandwidth: Q = center_freq / 5.0
        Q = center_freq_Hz / 5.0

        # Apply band-reject biquad filter to each channel
        filtered_channels = []
        for channel_data in data:
            filtered_channel = F.bandreject_biquad(
                waveform=channel_data.unsqueeze(0),  # Add batch dim
                sample_rate=fs,
                central_freq=center_freq_Hz,
                Q=Q
            ).squeeze(0)  # Remove batch dim
            filtered_channels.append(filtered_channel)

        filtered_data = torch.stack(filtered_channels)

        # Restore original shape if needed
        if len(original_shape) == 1:
            filtered_data = filtered_data.squeeze(0)

        return filtered_data

    except Exception as e:
        # If filter design fails, return original data
        print(f"Warning: Band-stop filter failed ({e}), returning original data")
        return data


def apply_augmentations_torch(
    data: torch.Tensor,
    augmentation_ranges: dict = None,
    num_augmentations: int = 2) -> tuple:
    """
    Apply random selection of augmentations to EEG signal on GPU.

    Args:
        data: EEG signal tensor (n_channels, n_samples)
        augmentation_ranges: Dictionary of augmentation parameter ranges
        num_augmentations: Number of augmentations to apply

    Returns:
        augmented_data: Augmented signal tensor (n_channels, n_samples)
        aug_info: List of (augmentation_name, parameter_value) tuples
    """
    if augmentation_ranges is None:
        augmentation_ranges = transformation_ranges

    # Available augmentations
    available_augs = ['amplitude_scale', 'time_shift', 'DC_shift', 'zero_masking', 'gaussian_noise', 'band_stop']

    # Randomly select augmentations (without replacement)
    num_augmentations = min(num_augmentations, len(available_augs))
    selected_indices = torch.randperm(len(available_augs))[:num_augmentations]
    selected_augs = [available_augs[i] for i in selected_indices]

    # Apply augmentations sequentially
    augmented_data = data.clone()
    aug_info = []

    for aug_name in selected_augs:
        if aug_name == 'amplitude_scale':
            low, high = augmentation_ranges['amplitude_scale']
            param = torch.rand(1, device=data.device).item() * (high - low) + low
            augmented_data = amplitude_scaling_torch(augmented_data, param)
            aug_info.append((aug_name, f'{param:.2f}x'))

        elif aug_name == 'time_shift':
            low, high = augmentation_ranges['time_shift_percent_twindow']
            param = torch.rand(1, device=data.device).item() * (high - low) + low
            augmented_data = time_shift_torch(augmented_data, param)
            aug_info.append((aug_name, f'{param*100:.1f}%'))

        elif aug_name == 'DC_shift':
            low, high = augmentation_ranges['DC_shift_µV']
            param = torch.rand(1, device=data.device).item() * (high - low) + low
            augmented_data = dc_shift_torch(augmented_data, param)
            aug_info.append((aug_name, f'{param:.2f}µV'))

        elif aug_name == 'zero_masking':
            low, high = augmentation_ranges['zero_masking_percent_twindow']
            param = torch.rand(1, device=data.device).item() * (high - low) + low
            augmented_data = zero_masking_torch(augmented_data, param)
            aug_info.append((aug_name, f'{param*100:.1f}%'))

        elif aug_name == 'gaussian_noise':
            low, high = augmentation_ranges['additive_gaussian_noise']
            param = torch.rand(1, device=data.device).item() * (high - low) + low
            augmented_data = additive_gaussian_noise_torch(augmented_data, param)
            aug_info.append((aug_name, f'σ={param:.2f}'))

        elif aug_name == 'band_stop':
            low, high = augmentation_ranges['band5hz_stop_Hzstart']
            param = torch.rand(1, device=data.device).item() * (high - low) + low
            augmented_data = band_stop_filter_torch(augmented_data, param)
            aug_info.append((aug_name, f'{param:.1f}Hz'))

    return augmented_data, aug_info


def create_augmented_pair_torch(
    data: torch.Tensor,
    augmentation_ranges: dict = None,
    num_augmentations: int = 2
) -> tuple:
    """
    Create two differently augmented versions of the same EEG sample on GPU (positive pair).
    """
    view1, info1 = apply_augmentations_torch(data, augmentation_ranges, num_augmentations)
    view2, info2 = apply_augmentations_torch(data, augmentation_ranges, num_augmentations)

    return view1, view2, info1, info2


def apply_augmentations_batch_torch(
    batch: torch.Tensor,
    augmentation_ranges: dict = None,
    num_augmentations: int = 2
) -> tuple:
    """
    Apply augmentations to a batch of EEG signals on GPU.
    """
    batch_size = batch.shape[0]
    augmented_samples = []
    aug_info_list = []

    for i in range(batch_size):
        aug_data, aug_info = apply_augmentations_torch(
            batch[i],
            augmentation_ranges,
            num_augmentations
        )
        augmented_samples.append(aug_data)
        aug_info_list.append(aug_info)

    augmented_batch = torch.stack(augmented_samples)
    return augmented_batch, aug_info_list


def create_augmented_pair_batch_torch(
    batch: torch.Tensor,
    augmentation_ranges: dict = None,
    num_augmentations: int = 2
) -> tuple:
    """
    Create two differently augmented versions for a batch of EEG samples on GPU.

    Args:
        batch: Batch of EEG signals (batch_size, n_channels, n_samples)
        augmentation_ranges: Dictionary of augmentation parameter ranges
        num_augmentations: Number of augmentations to apply per view

    Returns:
        view1_batch: First augmented views (batch_size, n_channels, n_samples)
        view2_batch: Second augmented views (batch_size, n_channels, n_samples)
        info1_list: List of augmentation info for view1
        info2_list: List of augmentation info for view2
    """
    view1_batch, info1_list = apply_augmentations_batch_torch(
        batch, augmentation_ranges, num_augmentations)
    view2_batch, info2_list = apply_augmentations_batch_torch(
        batch, augmentation_ranges, num_augmentations)

    return view1_batch, view2_batch, info1_list, info2_list
