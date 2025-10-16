"""
Create Challenge 2 Shards for Externalizing Factor Prediction

Creates sharded datasets from contrastChangeDetection task for Challenge 2.
Uses 4-second windows with 2-second stride as per the challenge starter kit.

Key differences from CRL pretraining:
- Single task: contrastChangeDetection only
- Window size: 4 seconds (vs 2s for CRL)
- Stride: 2 seconds
- Includes psychopathology factors: p_factor, attention, internalizing, externalizing
- Output shape: (129 channels, 400 samples) at 100Hz
"""

import pickle
import argparse
from pathlib import Path
from typing import Optional, List
import pandas as pd
import warnings
import sys
import math

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets.base import BaseConcatDataset
from eegdash import EEGChallengeDataset


# Subjects to exclude (from Challenge 2 starter kit)
EXCLUDED_SUBJECTS = [
    "NDARWV769JM7", "NDARME789TD2", "NDARUA442ZVF", "NDARJP304NK1",
    "NDARTY128YLU", "NDARDW550GU6", "NDARLD243KRE", "NDARUJ292JXV",
    "NDARBA381JGH"
]


def create_challenge2_shards_EEGChallenge(
    release: str = "R1",
    cache_dir: Path = PROJECT_ROOT / "database",
    savepath_root: Path = PROJECT_ROOT / "datasets",
    subjects: Optional[List[str]] = None,
    nb_subjects_per_shard: int = 10,
    window_length: float = 4.0,
    window_stride: float = 2.0,
    mini: bool = False,
    verbose: bool = True
) -> None:
    """
    Create Challenge 2 shards using EEGChallengeDataset.

    Note: Data is already preprocessed (0.5-50Hz bandpass, Cz reref, 100Hz).

    Args:
        release: Dataset release (e.g., "R5")
        cache_dir: Cache directory for EEGChallengeDataset
        savepath_root: Output directory for shards
        subjects: List of subject IDs (None = use all)
        nb_subjects_per_shard: Subjects per shard file
        window_length: Window length in seconds (default: 4.0s as per challenge)
        window_stride: Stride between windows in seconds (default: 2.0s)
        mini: Use mini release for testing
        verbose: Print progress
    """
    if verbose:
        print('=' * 70)
        print('  Challenge 2 Dataset Creation (EEGChallenge)')
        print('  Task: contrastChangeDetection')
        print('  Target: externalizing factor')
        print('=' * 70)

    savepath_root.mkdir(parents=True, exist_ok=True)

    # Load contrastChangeDetection task with psychopathology factors
    if verbose:
        print(f"\nLoading contrastChangeDetection task from {release}")

    dataset = EEGChallengeDataset(
        release=release,
        task="contrastChangeDetection",
        cache_dir=cache_dir,
        mini=mini,
        description_fields=[
            "subject", "session", "run", "task", "age", "sex",
            "p_factor", "attention", "internalizing", "externalizing"
        ]
    )

    if len(dataset.datasets) == 0:
        print(f"  ✗ No data found for {release}")
        return

    if verbose:
        print(f"  ✓ Loaded {len(dataset.datasets)} recordings")

    # Filter datasets: remove excluded subjects, too short recordings,
    # recordings with wrong channel count, or missing p_factor
    sfreq = 100
    min_samples = int(window_length * sfreq)

    filtered_datasets = [
        ds for ds in dataset.datasets
        if ds.description["subject"] not in EXCLUDED_SUBJECTS
        and ds.raw.n_times >= min_samples
        and len(ds.raw.ch_names) == 129
        and not math.isnan(ds.description.get("p_factor", float('nan')))
    ]

    if verbose:
        print(f"  ✓ After filtering: {len(filtered_datasets)} recordings")
        print(f"    (excluded {len(dataset.datasets) - len(filtered_datasets)} recordings)")

    # Create filtered BaseConcatDataset
    filtered_dataset = BaseConcatDataset(filtered_datasets)

    # Create fixed-length windows (4 seconds with 2 second stride)
    window_size_samples = int(window_length * sfreq)
    window_stride_samples = int(window_stride * sfreq)

    if verbose:
        print(f"\n  Creating windows:")
        print(f"    Window size: {window_length}s ({window_size_samples} samples)")
        print(f"    Stride: {window_stride}s ({window_stride_samples} samples)")

    windows = create_fixed_length_windows(
        filtered_dataset,
        window_size_samples=window_size_samples,
        window_stride_samples=window_stride_samples,
        drop_last_window=True,
    )

    if verbose:
        print(f"  ✓ Created {len(windows)} windows")

    # Get metadata
    metadata = windows.get_metadata()

    # Get list of subjects to process
    if subjects is None:
        subjects = metadata['subject'].unique().tolist()

    if verbose:
        print(f"\n  Total subjects: {len(subjects)}")
        print('=' * 70)

    # Extract data by subject and save shards
    challenge2_data = []
    shard_counter = 0

    for subj_idx, subject in enumerate(subjects):
        subject_mask = metadata['subject'] == subject
        subject_metadata = metadata[subject_mask]

        for idx in subject_metadata.index:
            X, _, _ = windows[idx]

            # Get description from the original dataset
            desc = subject_metadata.loc[idx]

            challenge2_data.append({
                'signal': X,  # Shape: (129, 400) for 4 seconds at 100Hz
                'subject': desc['subject'],
                'task': desc['task'],
                'session': desc.get('session', None),
                'run': desc.get('run', None),
                'age': desc.get('age', None),
                'sex': desc.get('sex', None),
                'p_factor': desc.get('p_factor', None),
                'attention': desc.get('attention', None),
                'internalizing': desc.get('internalizing', None),
                'externalizing': desc.get('externalizing', None)
            })

        if verbose and len(subject_metadata) > 0:
            ext_val = subject_metadata.iloc[0].get('externalizing', 'N/A')
            print(f"[{subj_idx+1}/{len(subjects)}] {subject}: {len(subject_metadata)} windows "
                  f"(externalizing={ext_val:.3f})" if isinstance(ext_val, (int, float))
                  else f"[{subj_idx+1}/{len(subjects)}] {subject}: {len(subject_metadata)} windows")

        # Save shard every nb_subjects_per_shard subjects
        if (subj_idx + 1) % nb_subjects_per_shard == 0 and challenge2_data:
            temp_df = pd.DataFrame(challenge2_data)
            shard_path = savepath_root / f'challenge2_data_shard_{shard_counter}_{release}.pkl'

            with open(shard_path, 'wb') as f:
                pickle.dump(temp_df, f)

            if verbose:
                print(f"\n{'='*70}")
                print(f"  ✓ Saved shard {shard_counter}: {len(temp_df)} windows")
                print(f"    File: {shard_path.name}")
                print(f"{'='*70}\n")

            shard_counter += 1
            challenge2_data = []

    # Save remaining data
    if challenge2_data:
        temp_df = pd.DataFrame(challenge2_data)
        shard_path = savepath_root / f'challenge2_data_shard_{shard_counter}_{release}.pkl'

        with open(shard_path, 'wb') as f:
            pickle.dump(temp_df, f)

        if verbose:
            print(f"\n{'='*70}")
            print(f"  ✓ Saved final shard {shard_counter}: {len(temp_df)} windows")
            print(f"    File: {shard_path.name}")
            print(f"{'='*70}")

    if verbose:
        print(f"\n  ✓ Complete! Total shards: {shard_counter + 1}")
        print('=' * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create Challenge 2 shards using EEGChallengeDataset"
    )

    parser.add_argument("--release", default="R5", help="Dataset release (default: R5)")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "database")
    parser.add_argument("--savepath-root", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--subjects-per-shard", type=int, default=10)
    parser.add_argument("--window-length", type=float, default=4.0, help="Window length in seconds")
    parser.add_argument("--window-stride", type=float, default=2.0, help="Window stride in seconds")
    parser.add_argument("--mini", action="store_true", help="Use mini release")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    create_challenge2_shards_EEGChallenge(
        release=args.release,
        cache_dir=args.cache_dir,
        savepath_root=args.savepath_root,
        nb_subjects_per_shard=args.subjects_per_shard,
        window_length=args.window_length,
        window_stride=args.window_stride,
        mini=args.mini,
        verbose=args.verbose
    )
