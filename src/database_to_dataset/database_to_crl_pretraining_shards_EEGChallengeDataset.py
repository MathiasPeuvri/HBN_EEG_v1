"""
Create Multi-Task Pretraining Shards for Contrastive Learning (10 sujet par shard car les shards avec plusieurs vues augmentés sont lourdes)

Creates sharded datasets from all 6 HBN tasks for CRL pretraining.
Unlike database_to_pretraining_shards.py (SuS only), this script aggregates
data from multiple tasks to learn more robust EEG representations.

Tasks included:
- RestingState (RS), surroundSupp (SuS), contrastChangeDetection (CCD), 
seqLearning8target (SL), symbolSearch (SyS), DespicableMe (MW)
"""

import pickle
import argparse
from pathlib import Path
from typing import Optional, List
import pandas as pd
import warnings
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets.base import BaseConcatDataset
from eegdash import EEGChallengeDataset


# All 6 HBN tasks for multi-task pretraining
DEFAULT_TASKS = ["RestingState", "surroundSupp", "contrastChangeDetection",
 "seqLearning8target", "symbolSearch", "DespicableMe"]


def create_crl_pretraining_shards_EEGChallenge(
    release: str = "R1",
    cache_dir: Path = PROJECT_ROOT / "database",
    savepath_root: Path = PROJECT_ROOT / "datasets",
    subjects: Optional[List[str]] = None,
    nb_subjects_per_shard: int = 10,
    tasks: List[str] = None,
    epoch_length: float = 2.0,
    overlap: float = 0.0,
    mini: bool = False,
    verbose: bool = True
) -> None:
    """
    Create CRL pretraining shards using EEGChallengeDataset.

    Note: Data is already preprocessed (0.5-50Hz bandpass, Cz reref, 100Hz).

    Args:
        release: Dataset release (e.g., "R5")
        cache_dir: Cache directory for EEGChallengeDataset
        savepath_root: Output directory for shards
        subjects: List of subject IDs (None = use all)
        nb_subjects_per_shard: Subjects per shard file
        tasks: Task names (None = use all 6 default tasks)
        epoch_length: Epoch length in seconds
        overlap: Overlap between epochs in seconds
        mini: Use mini release for testing
        verbose: Print progress
    """
    if verbose:
        print('=' * 70)
        print('  CRL Multi-Task Pretraining Dataset Creation (EEGChallenge)')
        print('=' * 70)

    if tasks is None:
        tasks = DEFAULT_TASKS

    savepath_root.mkdir(parents=True, exist_ok=True)

    # Load all tasks
    all_windows = []
    sfreq = 100
    window_stride = int((epoch_length - overlap) * sfreq)

    for task in tasks:
        if verbose:
            print(f"\nLoading task: {task}")

        dataset = EEGChallengeDataset(
            release=release,
            task=task,
            cache_dir=cache_dir,
            mini=mini,
            description_fields=["subject", "task", "run"]
        )

        if len(dataset.datasets) == 0:
            if verbose:
                print(f"  ✗ No data for {task}")
            continue

        windows = create_fixed_length_windows(
            dataset,
            window_size_samples=int(epoch_length * sfreq),
            window_stride_samples=window_stride,
            drop_last_window=True,
        )

        all_windows.append(windows)
        if verbose:
            print(f"  ✓ {len(windows)} windows created")

    # Merge all windows
    all_windows = BaseConcatDataset(all_windows)
    metadata = all_windows.get_metadata()

    if subjects is None:
        subjects = metadata['subject'].unique().tolist()

    if verbose:
        print(f"\n  Total subjects: {len(subjects)}")
        print(f"  Total windows:  {len(all_windows)}")
        print(f"  Epoch length:   {epoch_length}s")
        print(f"  Overlap:        {overlap}s")
        print('=' * 70)

    # Extract data by subject and save shards
    pretraining_data = []
    shard_counter = 0

    for subj_idx, subject in enumerate(subjects):
        subject_mask = metadata['subject'] == subject
        subject_metadata = metadata[subject_mask]

        for idx in subject_metadata.index:
            X, _, _ = all_windows[idx]

            pretraining_data.append({
                'signal': X,
                'subject': subject_metadata.loc[idx, 'subject'],
                'task': subject_metadata.loc[idx, 'task'],
                'run': subject_metadata.loc[idx, 'run'] if 'run' in subject_metadata.columns else None
            })

        if verbose and len(subject_metadata) > 0:
            print(f"[{subj_idx+1}/{len(subjects)}] {subject}: {len(subject_metadata)} epochs")

        # Save shard
        if (subj_idx + 1) % nb_subjects_per_shard == 0 and pretraining_data:
            temp_df = pd.DataFrame(pretraining_data)
            shard_path = savepath_root / f'crl_pretraining_data_shard_{shard_counter}_{release}.pkl'
            #shard_path = savepath_root / f'crl_pretraining_data_shard_{shard_counter}.pkl'

            with open(shard_path, 'wb') as f:
                pickle.dump(temp_df, f)

            if verbose:
                print(f"\n{'='*70}")
                print(f"  ✓ Saved shard {shard_counter}: {len(temp_df)} epochs")
                print(f"{'='*70}\n")

            shard_counter += 1
            pretraining_data = []

    # Save remaining
    if pretraining_data:
        temp_df = pd.DataFrame(pretraining_data)
        shard_path = savepath_root / f'crl_pretraining_data_shard_{shard_counter}_{release}.pkl'
        #shard_path = savepath_root / f'crl_pretraining_data_shard_{shard_counter}.pkl'

        with open(shard_path, 'wb') as f:
            pickle.dump(temp_df, f)

        if verbose:
            print(f"\n{'='*70}")
            print(f"  ✓ Saved final shard {shard_counter}: {len(temp_df)} epochs")
            print(f"{'='*70}")

    if verbose:
        print(f"\n  ✓ Complete! Total shards: {shard_counter + 1}")
        print('=' * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create CRL pretraining shards using EEGChallengeDataset"
    )

    parser.add_argument("--release", default="R5", help="Dataset release (default: R5)")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT_ROOT / "database")
    parser.add_argument("--savepath-root", type=Path, default=PROJECT_ROOT / "datasets")
    parser.add_argument("--subjects-per-shard", type=int, default=10)
    parser.add_argument("--tasks", nargs="+", default=None, help=f"Tasks: {', '.join(DEFAULT_TASKS)}")
    parser.add_argument("--epoch-length", type=float, default=2.0)
    parser.add_argument("--overlap", type=float, default=0.0)
    parser.add_argument("--mini", action="store_true", help="Use mini release")
    parser.add_argument("--verbose", action="store_true")

    args = parser.parse_args()

    create_crl_pretraining_shards_EEGChallenge(
        release=args.release,
        cache_dir=args.cache_dir,
        savepath_root=args.savepath_root,
        nb_subjects_per_shard=args.subjects_per_shard,
        tasks=args.tasks,
        epoch_length=args.epoch_length,
        overlap=args.overlap,
        mini=args.mini,
        verbose=args.verbose
    )
