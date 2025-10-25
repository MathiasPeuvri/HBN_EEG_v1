"""
Masking strategies for EEG data preprocessing
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import torch
from src.ML_pipeline_test import config


def create_timepoint_mask(batch_size, seq_len, mask_ratio=0.75, device=None):
    """Mask random timepoints (all channels at those timepoints)"""
    if device is None:
        device = config.DEVICE
    num_masked = int(seq_len * mask_ratio)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        masked_indices = torch.randperm(seq_len, device=device)[:num_masked]
        mask[i, masked_indices] = False
    return mask


def create_channel_mask(batch_size, num_channels, seq_len, mask_ratio=0.75, device=None):
    """Mask random channels (all timepoints for those channels)"""
    if device is None:
        device = config.DEVICE
    num_masked = int(num_channels * mask_ratio)
    mask = torch.ones(batch_size, num_channels, seq_len, dtype=torch.bool, device=device)
    for i in range(batch_size):
        masked_channels = torch.randperm(num_channels, device=device)[:num_masked]
        mask[i, masked_channels, :] = False
    return mask


def create_block_mask(batch_size, seq_len, mask_ratio=0.75, block_size=10, device=None):
    """Mask contiguous blocks of timepoints"""
    if device is None:
        device = config.DEVICE
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=device)
    num_masked_total = int(seq_len * mask_ratio)
    for i in range(batch_size):
        masked_count = 0
        while masked_count < num_masked_total:
            start_idx = torch.randint(0, seq_len, (1,), device=device).item()
            end_idx = min(start_idx + block_size, seq_len)
            mask[i, start_idx:end_idx] = False
            masked_count += (end_idx - start_idx)
    return mask


def create_and_apply_mask(x, mask_ratio=config.MASK_RATIO, mask_value=0, strategy='timepoint', block_size=10):
    """Create mask using specified strategy and apply to input data

    Args:
        strategy: 'timepoint', 'channel', or 'block'
    """
    batch_size, channels, seq_len = x.shape
    x_masked = x.clone()

    if strategy == 'timepoint':
        mask = create_timepoint_mask(batch_size, seq_len, mask_ratio, device=x.device)
        for b in range(batch_size):
            x_masked[b, :, ~mask[b]] = mask_value

    elif strategy == 'channel':
        mask = create_channel_mask(batch_size, channels, seq_len, mask_ratio, device=x.device)
        x_masked[~mask] = mask_value

    elif strategy == 'block':
        mask = create_block_mask(batch_size, seq_len, mask_ratio, block_size, device=x.device)
        for b in range(batch_size):
            x_masked[b, :, ~mask[b]] = mask_value

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return x_masked, mask


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    test_data = torch.ones(1, 129, 200)
    mask_ratio = 0.25

    fig, axes = plt.subplots(4, 1, figsize=(12, 8))

    # Original
    axes[0].imshow(test_data[0].numpy(), aspect='auto', cmap='viridis')
    axes[0].set_title('Original Data')
    axes[0].set_ylabel('Channels')

    # Timepoint masking
    masked_tp, _ = create_and_apply_mask(test_data, mask_ratio, strategy='timepoint')
    axes[1].imshow(masked_tp[0].numpy(), aspect='auto', cmap='viridis')
    axes[1].set_title(f'Timepoint Masking (mask_ratio={mask_ratio})')
    axes[1].set_ylabel('Channels')

    # Channel masking
    masked_ch, _ = create_and_apply_mask(test_data, mask_ratio, strategy='channel')
    axes[2].imshow(masked_ch[0].numpy(), aspect='auto', cmap='viridis')
    axes[2].set_title(f'Channel Masking (mask_ratio={mask_ratio})')
    axes[2].set_ylabel('Channels')

    # Block masking
    masked_bl, _ = create_and_apply_mask(test_data, mask_ratio, strategy='block', block_size=10)
    axes[3].imshow(masked_bl[0].numpy(), aspect='auto', cmap='viridis')
    axes[3].set_title(f'Block Masking (mask_ratio={mask_ratio}, block_size=10)')
    axes[3].set_ylabel('Channels')
    axes[3].set_xlabel('Timepoints')
    axes[3].set_ylim(0, 129)

    plt.tight_layout()
    plt.show()