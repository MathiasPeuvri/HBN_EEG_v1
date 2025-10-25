"""
Sequential shard dataset for downstream tasks (classification/regression)
"""
import glob
import hashlib
import pickle
import random
import re
import numpy as np
import pandas as pd
import torch
from pathlib import Path
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
                 data_format: str = "standard"):
        """
        Args:
            shard_pattern: Glob pattern for shard files (e.g., "downstream_data_shard_*.pkl")
            task_type: 'classification' or 'regression' (uses config default if None)
            train_split: Proportion of shards for training
            is_train: Whether this is training or validation set
            seed: Random seed for reproducibility
            data_format: Format of shard data:
                'standard' (default) - standard shards with 'signal' column, no conversion
                'v1' - Maxime's format with winwdows/vals, converted to signal/response_time
                'v2_windowed' - Maxime's format with temporal localization augmentation
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
            # Detect releases from shard filenames and load corresponding participants.tsv
            self.psycho_factor_map = self._load_participants_for_shards()
            print(f"Loaded psychopathology factor '{config.TARGET_COLUMN}' for {len(self.psycho_factor_map)} participants")
        
        # For classification, build global label mapping from all shards
        if task_type == 'classification':
            self.label_map = self._build_global_label_map()
            print(f"Global label mapping: {self.label_map}")
        else:
            self.label_map = None
        
        # Deterministic train/val split using stable hashing
        self._split_shards()

    def _detect_releases_from_shards(self) -> set:
        """
        Detect release identifiers from shard filenames.

        Looks for patterns like:
        - challenge2_data_shard_0_R1.pkl -> R1
        - pretraining_data_shard_5_R2.pkl -> R2

        Returns:
            set: Set of release identifiers (e.g., {'R1', 'R2'})
        """
        releases = set()

        # Pattern to match _R<number> in filenames
        release_pattern = re.compile(r'_R(\d+)(?:\.pkl)?$')

        for shard_file in self.shard_files:
            filename = Path(shard_file).name
            match = release_pattern.search(filename)
            if match:
                release_num = match.group(1)
                releases.add(f"R{release_num}")

        return releases

    def _load_participants_for_shards(self) -> Dict[str, float]:
        """
        Load participants.tsv files for all releases detected in shards.

        Returns:
            dict: Mapping from subject_id to psychopathology factor value
        """
        releases = self._detect_releases_from_shards()

        if not releases:
            print("Warning: No release identifiers found in shard filenames. Using default participants.tsv")
            if config.PARTICIPANTS_TSV_PATH and config.PARTICIPANTS_TSV_PATH.exists():
                releases = {'R1'}  # Default fallback
            else:
                raise FileNotFoundError("No participants.tsv found and no release detected from shards")

        print(f"Detected releases from shards: {sorted(releases)}")

        # Load and merge participants.tsv from all detected releases
        psycho_factor_map = {}

        for release in sorted(releases):
            try:
                participants_path = config.get_participants_tsv_for_release(release)
                print(f"  Loading {release}: {participants_path.relative_to(config.PROJECT_ROOT)}")

                participants_df = pd.read_csv(participants_path, sep='\t')

                # Create mapping from subject_id to factor value
                for _, row in participants_df.iterrows():
                    # Handle both with and without 'sub-' prefix
                    subject_id = row['participant_id'].replace('sub-', '')
                    if config.TARGET_COLUMN in row:
                        psycho_factor_map[subject_id] = row[config.TARGET_COLUMN]

            except (FileNotFoundError, ValueError) as e:
                print(f"  Warning: Could not load participants.tsv for {release}: {e}")
                continue

        if not psycho_factor_map:
            raise ValueError(
                f"Could not load psychopathology factor '{config.TARGET_COLUMN}' for any detected releases: {releases}"
            )

        return psycho_factor_map

    def _split_shards(self):
        """Split shards into train/val sets based on R2 naming convention"""
        # R2 files go to validation, others to train
        train_shards = [f for f in self.shard_files if 'R2' not in f]
        val_shards = [f for f in self.shard_files if 'R2' in f]

        self.active_shards = train_shards if self.is_train else val_shards

        if not self.active_shards:
            raise ValueError(f"No {'train' if self.is_train else 'validation'} shards found. "
                            f"Train shards: {len(train_shards)}, Val shards: {len(val_shards)}")

        print(f"{'Train' if self.is_train else 'Val'} downstream dataset: "
              f"{len(self.active_shards)}/{len(self.shard_files)} shards "
              f"(Train: {len(train_shards)}, Val: {len(val_shards)})")
    
    def _build_global_label_map(self) -> Dict[Any, int]:
        """Build global label mapping by scanning all shards"""
        unique_labels = set()

        for shard_file in self.shard_files:
            with open(shard_file, 'rb') as f:
                shard_data = pickle.load(f)

            # Apply appropriate format conversion (skip for standard format)
            if self.data_format == "v1":
                shard_data = convert_maximv1_format(shard_data)
            elif self.data_format == "v2_windowed":
                shard_data = convert_maximv2_format_with_window_augmentation(shard_data)
            # else: standard format, no conversion needed

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
                    # Handle case where subject_id might be a Series instead of a scalar
                    if isinstance(subject_id, pd.Series):
                        subject_id = subject_id.iloc[0] if len(subject_id) > 0 else subject_id.values[0]
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

            # Apply appropriate format conversion (skip for standard format)
            if self.data_format == "v1":
                shard_data = convert_maximv1_format(shard_data)
            elif self.data_format == "v2_windowed":
                shard_data = convert_maximv2_format_with_window_augmentation(pd.DataFrame(shard_data))
            # else: standard format, no conversion needed

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

                # Validate signal shape - accept both standard (200) and challenge 2 (400) window sizes
                if signal.shape[0] != config.NUM_CHANNELS:
                    continue
                if signal.shape[1] not in [config.POSTTRAINING_SEQ_LEN, config.CHALLENGE2_SEQ_LEN]:
                    # Accept 200 (standard) or 400 (challenge 2) samplepoints
                    continue

                # Extract target
                if self.is_psychopathology_target:
                    subject_id = row['subject']
                    # Handle case where subject_id might be a Series instead of a scalar
                    if isinstance(subject_id, pd.Series):
                        subject_id = subject_id.iloc[0] if len(subject_id) > 0 else subject_id.values[0]
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