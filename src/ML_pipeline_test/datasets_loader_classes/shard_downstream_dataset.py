"""
Sequential shard dataset for downstream tasks (classification/regression)
"""
import glob
import hashlib
import pickle
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import IterableDataset
from typing import Iterator, Tuple, Optional, Dict, Any
from .. import config
from ...test_pkl_maxime import convert_maximv2_format_with_window_augmentation


def convert_maximv1_format(data):
    """Convert eval format (winwdows/vals) to standard format (signal/response_time)."""
    # Convert to DataFrame if needed
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Check if already in expected format
    if 'signal' in data.columns or 'winwdows' not in data.columns:
        return data

    # Unfold: 1 row with N windows -> N rows with 1 window each
    rows = [{'signal': row['winwdows'][i], 'response_time': row['vals'][i, 0],
             'p_factor': row['vals'][i, 1], 'attention': row['vals'][i, 2],
             'internalizing': row['vals'][i, 3], 'externalizing': row['vals'][i, 4],
             'subject': row['subject']}
            for _, row in data.iterrows() for i in range(len(row['winwdows']))]

    return pd.DataFrame(rows)


class SequentialShardDownstreamDataset(IterableDataset):
    """Sequential shard dataset for downstream tasks with memory efficiency"""
    
    def __init__(self, shard_pattern: str, task_type: str = None,
                 train_split: float = 0.8, is_train: bool = True, seed: int = 42,
                 data_format: str = "v1"):
        """
        Args:
            shard_pattern: Glob pattern for shard files (e.g., "downstream_data_shard_*.pkl")
            task_type: 'classification' or 'regression' (uses config default if None)
            train_split: Proportion of shards for training
            is_train: Whether this is training or validation set
            seed: Random seed for reproducibility
            data_format: for chall 1 windowing from maxime files,
                'v1' (standard) or 'v2_windowed' (temporal localization)
        """
        if task_type is None:
            task_type = config.TASK_TYPE
        self.task_type = task_type
        self.data_format = data_format

        # Find all shard files
        self.shard_files = sorted(glob.glob(shard_pattern))
        if not self.shard_files:
            raise ValueError(f"No shard files found matching pattern: {shard_pattern}")
        
        self.train_split = train_split
        self.is_train = is_train
        self.seed = seed
        
        # Check if target is a psychopathology factor
        self.is_psychopathology_target = config.TARGET_COLUMN in config.PSYCHOPATHOLOGY_FACTORS
        if self.is_psychopathology_target:
            # Load participants data for psychopathology factor lookup
            self.participants_df = pd.read_csv(config.PARTICIPANTS_TSV_PATH, sep='\t')
            # Create mapping from subject_id to factor value
            self.psycho_factor_map = {}
            for _, row in self.participants_df.iterrows():
                # Handle both with and without 'sub-' prefix
                subject_id = row['participant_id'].replace('sub-', '')
                self.psycho_factor_map[subject_id] = row[config.TARGET_COLUMN]
            print(f"Loaded psychopathology factor '{config.TARGET_COLUMN}' for {len(self.psycho_factor_map)} participants")
        
        # For classification, build global label mapping from all shards
        if task_type == 'classification':
            self.label_map = self._build_global_label_map()
            print(f"Global label mapping: {self.label_map}")
        else:
            self.label_map = None
        
        # Deterministic train/val split using stable hashing
        self._split_shards()
        
    def _get_stable_hash(self, filename: str) -> int:
        """Get deterministic hash for filename"""
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
            # If no shards assigned due to small dataset, use fallback
            if not self.is_train and self.shard_files:
                # If no val shards, use last shard
                self.active_shards = [self.shard_files[-1]]
            elif self.is_train and self.shard_files:
                # If no train shards, use first shard
                self.active_shards = [self.shard_files[0]]
        
        print(f"{'Train' if self.is_train else 'Val'} downstream dataset: "
              f"{len(self.active_shards)} shards")
    
    def _build_global_label_map(self) -> Dict[Any, int]:
        """Build global label mapping by scanning all shards"""
        unique_labels = set()
        
        for shard_file in self.shard_files:
            with open(shard_file, 'rb') as f:
                shard_data = pickle.load(f)

            # Apply appropriate format conversion
            if self.data_format == "v1":
                shard_data = convert_maximv1_format(shard_data)
            else:  # v2_windowed
                shard_data = convert_maximv2_format_with_window_augmentation(shard_data)
            df_filtered = shard_data.copy()

            # Filter data based on TARGET_EVENTS if specified (skip for psychopathology factors)
            if config.TARGET_EVENTS is not None and not self.is_psychopathology_target:
                mask = df_filtered[config.TARGET_COLUMN].isin(config.TARGET_EVENTS)
                df_filtered = df_filtered[mask]

            # Collect unique labels for this shard
            if self.is_psychopathology_target:
                # For psychopathology factors, get unique subject IDs and map to factor values
                for _, row in df_filtered.iterrows():
                    subject_id = row['subject']
                    if subject_id in self.psycho_factor_map:
                        unique_labels.add(self.psycho_factor_map[subject_id])
            else:
                # Use column from data directly
                unique_labels.update(df_filtered[config.TARGET_COLUMN].unique())
        
        # Create sorted label mapping for consistency
        sorted_labels = sorted(unique_labels)
        return {label: i for i, label in enumerate(sorted_labels)}
    
    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """Iterate through all shards sequentially, yielding (signal, label) pairs"""
        # Create local random generator for this epoch
        rng = random.Random(self.seed)
        shard_order = self.active_shards.copy() 
        rng.shuffle(shard_order) # Shuffle shard order for this epoch

        for shard_file in shard_order:
            # Load shard with context manager for memory cleanup
            with open(shard_file, 'rb') as f:
                shard_data = pickle.load(f)

            if self.data_format == "v1": # chall 1 standard
                shard_data = convert_maximv1_format(shard_data)
            else:  # inwindow_rtidx_augmentation approach
                shard_data = convert_maximv2_format_with_window_augmentation(pd.DataFrame(shard_data))
            df_filtered = shard_data.copy()

            # Filter data based on TARGET_EVENTS if specified (skip for psychopathology factors)
            if config.TARGET_EVENTS is not None and not self.is_psychopathology_target:
                mask = df_filtered[config.TARGET_COLUMN].isin(config.TARGET_EVENTS)
                df_filtered = df_filtered[mask]

            # Extract signals and labels - YIELD DIRECTLY (no accumulation)
            # Shuffle indices instead of data to save RAM
            indices = np.arange(len(df_filtered))
            rng.shuffle(indices)

            for idx in indices:
                row = df_filtered.iloc[idx]
                signal = row['signal']

                if signal.shape != (config.NUM_CHANNELS, config.POSTTRAINING_SEQ_LEN):
                    continue

                # Extract target
                if self.is_psychopathology_target:
                    subject_id = row['subject']
                    if subject_id not in self.psycho_factor_map:
                        continue
                    target_value = self.psycho_factor_map[subject_id]
                else:
                    target_value = row[config.TARGET_COLUMN]

                # Skip invalid
                if self.task_type == 'regression' and pd.isna(target_value):
                    continue
                if self.task_type == 'classification' and target_value == 'no_response':
                    continue

                # Map label
                target = self.label_map[target_value] if self.task_type == 'classification' else target_value

                # Yield immediately
                signal_tensor = torch.tensor(signal.astype(np.float32), dtype=torch.float32)
                label_tensor = torch.tensor(target, dtype=torch.long if self.task_type == 'classification' else torch.float32)
                yield signal_tensor, label_tensor

            # Explicit cleanup to free memory immediately
            del shard_data, df_filtered