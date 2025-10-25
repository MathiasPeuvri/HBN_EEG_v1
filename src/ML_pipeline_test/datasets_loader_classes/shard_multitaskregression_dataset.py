"""Multi-task dataset for all 4 psychopathology factors"""
import torch
from .shard_downstream_dataset import SequentialShardDownstreamDataset
from .. import config


class MultiTaskDownstreamDataset(SequentialShardDownstreamDataset):
    """Extends SequentialShardDownstreamDataset to return all 4 psycho factors"""

    def __init__(self, shard_pattern: str, train_split: float = 0.8,
                 is_train: bool = True, seed: int = 42, data_format: str = "standard"):
        # All 4 factors we want to predict
        self.tasks = ['p_factor', 'externalizing', 'internalizing', 'attention']

        # Initialize with placeholder (parent expects TARGET_COLUMN)
        config.TARGET_COLUMN = 'p_factor'
        config.TASK_TYPE = 'regression'

        super().__init__(shard_pattern, task_type='regression',
                        train_split=train_split, is_train=is_train,
                        seed=seed, data_format=data_format)

    def _load_participants_for_shards(self):
        """Load ALL 4 psychopathology factors instead of just one"""
        import pandas as pd
        releases = self._detect_releases_from_shards()

        print(f"Detected releases from shards: {sorted(releases)}")

        # Map: subject_id -> {task: value}
        psycho_factors_map = {}

        for release in sorted(releases):
            try:
                participants_path = config.get_participants_tsv_for_release(release)
                print(f"  Loading {release}: {participants_path.relative_to(config.PROJECT_ROOT)}")

                df = pd.read_csv(participants_path, sep='\t')

                for _, row in df.iterrows():
                    subject_id = row['participant_id'].replace('sub-', '')

                    # Load all 4 factors
                    factors = {}
                    for task in self.tasks:
                        if task in row and not pd.isna(row[task]):
                            factors[task] = row[task]

                    # Only add if all 4 factors present
                    if len(factors) == 4:
                        psycho_factors_map[subject_id] = factors

            except Exception as e:
                print(f"  Warning: Could not load {release}: {e}")
                continue

        print(f"Loaded all 4 factors for {len(psycho_factors_map)} participants")
        return psycho_factors_map

    def __iter__(self):
        """Yield (signal, targets_dict) where targets_dict has all 4 factors"""
        import random
        import pickle
        import numpy as np
        import pandas as pd
        from ...test_pkl_maxime import convert_maximv2_format_with_window_augmentation
        from .shard_downstream_dataset import convert_maximv1_format

        rng = random.Random(self.seed)
        shard_order = self.active_shards.copy()
        rng.shuffle(shard_order)

        for shard_file in shard_order:
            with open(shard_file, 'rb') as f:
                shard_data = pickle.load(f)

            # Apply format conversion
            if self.data_format == "v1":
                shard_data = convert_maximv1_format(shard_data)
            elif self.data_format == "v2_windowed":
                shard_data = convert_maximv2_format_with_window_augmentation(pd.DataFrame(shard_data))

            df_filtered = shard_data.copy()

            # Shuffle
            indices = np.arange(len(df_filtered))
            rng.shuffle(indices)

            for idx in indices:
                row = df_filtered.iloc[idx]
                signal = row['signal']

                # Validate signal
                if signal.shape[0] != config.NUM_CHANNELS:
                    continue
                if signal.shape[1] not in [config.POSTTRAINING_SEQ_LEN, config.CHALLENGE2_SEQ_LEN]:
                    continue

                # Get subject ID
                subject_id = row['subject']
                if isinstance(subject_id, pd.Series):
                    subject_id = subject_id.iloc[0] if len(subject_id) > 0 else subject_id.values[0]

                # Skip if subject not in psycho factor map
                if subject_id not in self.psycho_factor_map:
                    continue

                # Get all 4 factors for this subject (now psycho_factor_map is a dict of dicts)
                subject_factors = self.psycho_factor_map.get(subject_id)
                if subject_factors is None or len(subject_factors) != 4:
                    continue

                # Convert to tensors
                signal_tensor = torch.tensor(signal.astype(np.float32), dtype=torch.float32)
                targets_tensor = {task: torch.tensor(val, dtype=torch.float32)
                                 for task, val in subject_factors.items()}

                yield signal_tensor, targets_tensor