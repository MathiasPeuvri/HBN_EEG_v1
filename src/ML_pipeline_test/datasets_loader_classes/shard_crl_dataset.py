"""
Shard-based Contrastive Learning Dataset

Loads multi-task EEG shards and wraps them for contrastive learning.
Supports memory-efficient training on large datasets by loading shards sequentially.

Note: Unified version with optional metadata - no separate classes needed.
"""

import glob
import hashlib
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset
from typing import Iterator
from ..contrastive_learning.augmentations import create_augmented_pair
from ..contrastive_learning.config import transformation_ranges


class ContrastiveShardDataset(IterableDataset):
    """
    Iterable dataset for contrastive learning on sharded EEG data.

    Unified version with optional metadata support. Loads shards sequentially
    and generates augmented pairs on-the-fly for memory efficiency.

    Args:
        shard_pattern: Glob pattern for shard files (e.g., "crl_pretraining_data_shard_*.pkl")
        train_split: Proportion of shards for training (default: 0.8)
        is_train: Whether this is training or validation set
        augmentation_ranges: Dictionary of augmentation parameter ranges
        num_augmentations: Number of augmentations per view (default: 2)
        return_metadata: Whether to return metadata (default: False)
        metadata_keys: List of metadata keys to return (e.g., ['subject', 'task'])
        seed: Random seed for reproducibility
    """

    def __init__(
        self,
        shard_pattern: str,
        train_split: float = 0.8,
        is_train: bool = True,
        augmentation_ranges: dict = None,
        num_augmentations: int = 2,
        return_metadata: bool = False,
        metadata_keys: list = None,
        seed: int = 42
    ):
        """Initialize contrastive shard dataset."""
        self.shard_files = sorted(glob.glob(shard_pattern))

        if not self.shard_files:
            raise ValueError(f"No shard files found matching pattern: {shard_pattern}")

        self.train_split = train_split
        self.is_train = is_train
        self.augmentation_ranges = augmentation_ranges or transformation_ranges
        self.num_augmentations = num_augmentations
        self.return_metadata = return_metadata
        self.metadata_keys = metadata_keys or ['subject', 'task']
        self.seed = seed

        self._split_shards()

    def _get_stable_hash(self, filename: str) -> int:
        """Get deterministic hash for filename."""
        return int(hashlib.md5(filename.encode()).hexdigest(), 16)

    def _split_shards(self):
        """Split shards into train/val sets deterministically."""
        train_shards = []
        val_shards = []

        for shard_file in self.shard_files:
            # Use stable hash for consistent splitting
            shard_hash = self._get_stable_hash(shard_file)
            if (shard_hash % 100) < (self.train_split * 100):
                train_shards.append(shard_file)
            else:
                val_shards.append(shard_file)

        self.active_shards = train_shards if self.is_train else val_shards

        # If no val shards due to small dataset, use last shard
        if not self.active_shards:
            if not self.is_train and self.shard_files:
                self.active_shards = [self.shard_files[-1]]

        print(f"{'Train' if self.is_train else 'Val'} CRL dataset: "
              f"{len(self.active_shards)} shards")

    def __iter__(self) -> Iterator[tuple]:
        """
        Iterate through all shards, generating augmented pairs.

        Yields:
            If return_metadata=False: (view1, view2)
            If return_metadata=True: (view1, view2, metadata_dict)
        """
        # Create local random generator for this epoch
        rng = random.Random(self.seed)

        # Shuffle shard order for this epoch
        shard_order = self.active_shards.copy()
        rng.shuffle(shard_order)

        for shard_file in shard_order:
            # Load shard
            with open(shard_file, 'rb') as f:
                shard_data = pickle.load(f)

            # Extract signals and optionally metadata from DataFrame
            if isinstance(shard_data, pd.DataFrame):
                signals = [row['signal'] for _, row in shard_data.iterrows()]
                samples = np.array(signals, dtype=np.float32)

                # Extract metadata if requested
                metadata = {}
                if self.return_metadata:
                    for key in self.metadata_keys:
                        if key in shard_data.columns:
                            metadata[key] = shard_data[key].values
            else:
                # Legacy numpy array format (no metadata)
                samples = shard_data.astype(np.float32)
                metadata = {}

            # Validate shape
            assert samples.ndim == 3, f"Expected 3D array, got shape {samples.shape}"

            # Create indices and shuffle within shard
            indices = list(range(len(samples)))
            rng.shuffle(indices)

            # Generate augmented pairs for each sample
            for idx in indices:
                signal = samples[idx]

                # Create augmented pair
                view1, view2, _, _ = create_augmented_pair(
                    signal,
                    self.augmentation_ranges,
                    self.num_augmentations
                )

                # Convert to tensors
                view1_tensor = torch.tensor(view1, dtype=torch.float32)
                view2_tensor = torch.tensor(view2, dtype=torch.float32)

                # Yield with or without metadata
                if self.return_metadata and metadata:
                    metadata_dict = {}
                    for key, values in metadata.items():
                        metadata_dict[key] = values[idx]
                    yield (view1_tensor, view2_tensor, metadata_dict)
                else:
                    yield (view1_tensor, view2_tensor)

            # Explicit cleanup
            del shard_data, samples
