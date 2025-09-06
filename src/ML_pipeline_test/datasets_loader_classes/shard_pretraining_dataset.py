"""
Sequential shard dataset for memory-efficient training
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
from .. import config


class SequentialShardDataset(IterableDataset):
    """Sequential shard dataset for memory-efficient training"""
    
    def __init__(self, shard_pattern: str, train_split: float = 0.8, 
                 is_train: bool = True, seed: int = 42):
        """
        Args:
            shard_pattern: Glob pattern for shard files (e.g., "data_shard_*.pkl")
            train_split: Proportion of shards for training
            is_train: Whether this is training or validation set
            seed: Random seed for reproducibility
        """
        # Find all shard files - support both 'batch' and 'shard' naming
        self.shard_files = sorted(glob.glob(shard_pattern))
        if not self.shard_files:
            # Fallback to batch pattern for backward compatibility
            batch_pattern = shard_pattern.replace('shard', 'batch')
            self.shard_files = sorted(glob.glob(batch_pattern))
        
        if not self.shard_files:
            raise ValueError(f"No shard files found matching pattern: {shard_pattern}")
        
        self.train_split = train_split
        self.is_train = is_train
        self.seed = seed
        
        # Deterministic train/val split using stable hashing
        self._split_shards()
        
    def _get_stable_hash(self, filename: str) -> int:
        """Get deterministic hash for filename"""
        # Use hashlib instead of hash() for cross-process consistency
        return int(hashlib.md5(filename.encode()).hexdigest(), 16)
    
    def _split_shards(self):
        """Split shards into train/val sets deterministically"""
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
        
        if not self.active_shards:
            # If no val shards due to small dataset, use last shard
            if not self.is_train and self.shard_files:
                self.active_shards = [self.shard_files[-1]]
        
        print(f"{'Train' if self.is_train else 'Val'} dataset: "
              f"{len(self.active_shards)} shards")
    
    def __iter__(self) -> Iterator[torch.Tensor]:
        """Iterate through all shards sequentially"""
        # Create local random generator for this epoch
        rng = random.Random(self.seed)
        
        # Shuffle shard order for this epoch
        shard_order = self.active_shards.copy()
        rng.shuffle(shard_order)
        
        for shard_file in shard_order:
            # Load shard with context manager for memory cleanup
            with open(shard_file, 'rb') as f:
                shard_data = pickle.load(f)
            
            # Handle both DataFrame and array formats
            if isinstance(shard_data, pd.DataFrame):
                # Extract signals from DataFrame (following SSL_Dataset pattern)
                signals = [row['signal'] for _, row in shard_data.iterrows()]
                samples = np.array(signals, dtype=np.float32)
            else:
                # Legacy numpy array format
                samples = shard_data.astype(np.float32)
            
            # Validate shape (following SSL_Dataset assertions)
            assert samples.shape[1] == config.NUM_CHANNELS
            assert samples.shape[2] == config.PRETRAINING_SEQ_LEN
            
            # Create indices and shuffle within shard
            indices = list(range(len(samples)))
            rng.shuffle(indices)
            
            # Yield samples as tensors
            for idx in indices:
                yield torch.tensor(samples[idx], dtype=torch.float32)
            
            # Explicit cleanup to free memory immediately
            del shard_data, samples