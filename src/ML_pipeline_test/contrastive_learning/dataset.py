"""
Contrastive Learning Dataset for EEG

PyTorch Dataset that generates augmented pairs for contrastive learning.
Each sample returns two differently augmented views of the same EEG epoch.

Note: Unified version with optional metadata - avoids code duplication.
"""

import torch
import numpy as np
from torch.utils.data import Dataset
from .augmentations import create_augmented_pair
from .config import transformation_ranges


class ContrastiveEEGDataset(Dataset):
    """
    Unified contrastive dataset with optional metadata support.

    For each EEG epoch, generates two differently augmented versions (positive pair).
    Optionally returns metadata (subject, task, etc.) for downstream tasks.

    The metadata overhead is negligible (just strings), so we always use this unified
    version to avoid code duplication. If no metadata provided, returns empty dict.

    Args:
        signals: numpy array of EEG signals (n_epochs, n_channels, n_samples)
                 or list of numpy arrays
        metadata: Dictionary with keys like 'subject', 'task', 'run' (optional)
        augmentation_ranges: Dictionary of augmentation parameter ranges
        num_augmentations: Number of augmentations to apply per view (default: 2)
        return_metadata: Whether to return metadata in __getitem__ (default: False)

    Returns:
        If return_metadata=False: (view1, view2)
        If return_metadata=True: (view1, view2, metadata_dict)

    Example:
        >>> # Without metadata (for standard pretraining)
        >>> dataset = ContrastiveEEGDataset(signals)
        >>> view1, view2 = dataset[0]
        >>>
        >>> # With metadata (for downstream tasks needing subject info)
        >>> dataset = ContrastiveEEGDataset(signals, metadata={'subject': subjects}, return_metadata=True)
        >>> view1, view2, meta = dataset[0]
    """

    def __init__(
        self,
        signals: np.ndarray,
        metadata: dict = None,
        augmentation_ranges: dict = None,
        num_augmentations: int = 2,
        return_metadata: bool = False
    ):
        """
        Initialize contrastive dataset.

        Args:
            signals: EEG data (n_epochs, n_channels, n_samples)
            metadata: Optional dict with 'subject', 'task', etc. arrays
            augmentation_ranges: Augmentation parameter ranges
            num_augmentations: Number of augmentations per view
            return_metadata: Whether to return metadata in __getitem__
        """
        # Handle list input (convert to array)
        if isinstance(signals, list):
            signals = np.array(signals, dtype=np.float32)
        else:
            signals = signals.astype(np.float32)

        self.signals = signals
        self.metadata = metadata or {}
        self.augmentation_ranges = augmentation_ranges or transformation_ranges
        self.num_augmentations = num_augmentations
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        """Return number of samples in dataset."""
        return len(self.signals)

    def __getitem__(self, idx: int) -> tuple:
        """
        Get augmented pair (optionally with metadata).

        Args:
            idx: Index of sample to retrieve

        Returns:
            If return_metadata=False: (view1, view2)
            If return_metadata=True: (view1, view2, metadata_dict)
        """
        signal = self.signals[idx]

        # Create two differently augmented versions
        view1, view2, _, _ = create_augmented_pair(
            signal,
            self.augmentation_ranges,
            self.num_augmentations
        )

        # Convert to torch tensors
        view1 = torch.FloatTensor(view1)
        view2 = torch.FloatTensor(view2)

        if self.return_metadata:
            # Extract metadata for this sample
            metadata_dict = {}
            for key, values in self.metadata.items():
                metadata_dict[key] = values[idx]
            return view1, view2, metadata_dict
        else:
            return view1, view2
