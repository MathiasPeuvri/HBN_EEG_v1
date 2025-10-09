"""
Create Multi-Task Pretraining Shards for Contrastive Learning (10 sujet par shard car les shards avec plusieurs vues augmentés sont lourdes)

Creates sharded datasets from all 6 HBN tasks for CRL pretraining.
Unlike database_to_pretraining_shards.py (SuS only), this script aggregates
data from multiple tasks to learn more robust EEG representations.

Tasks included:
- RestingState (RS)
- surroundSupp (SuS) - passive
- contrastChangeDetection (CCD) - active
- seqLearning8target (SL)
- symbolSearch (SyS)
- DespicableMe (MW) - movie watching
"""

import pickle
import sys
import argparse
from pathlib import Path
from typing import Optional, List
import numpy as np
import pandas as pd
import mne
import warnings

mne.set_log_level('ERROR')
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.insert(0, '/home/mts/HBN_EEG_v1')
from src.preprocessing.epoching import segment_continuous_numpy
from src.preprocessing.filters import preprocess_data
from src.loader.simple_loader import SimpleConfig, SimpleHBNLoader


# All 6 HBN tasks for multi-task pretraining
DEFAULT_TASKS = [
    "RestingState",
    "surroundSupp",
    "contrastChangeDetection",
    "seqLearning8target",
    "symbolSearch",
    "DespicableMe"
]


def create_crl_pretraining_shards(
    dataset_name: str = "R1_L100",
    database_root: Path = Path("/home/mts/HBN_EEG_v1/database/"),
    savepath_root: Path = Path("/home/mts/HBN_EEG_v1/datasets/"),
    subjects: Optional[List[str]] = None,
    nb_subjects_per_shard: int = 10,
    tasks: List[str] = None,
    runs: List[int] = [1, 2],
    epoch_length: float = 2.0,
    overlap: float = 0.0,
    verbose: bool = True
) -> None:
    """
    Create CRL pretraining data shards from multi-task EEG database.

    Args:
        dataset_name: Dataset name (e.g., "R1_L100")
        database_root: Root directory of EEG database
        savepath_root: Output directory for shards
        subjects: List of subject IDs (None = use all available)
        nb_subjects_per_shard: Number of subjects per shard file
        tasks: List of task names to include (None = use all 6 default tasks)
        runs: List of run numbers to process for each task
        epoch_length: Length of each epoch in seconds (default: 2.0)
        overlap: Overlap between epochs in seconds (default: 0.0)
        verbose: Print progress information

    Returns:
        None (saves shard files to disk)

    Example:
        >>> create_crl_pretraining_shards(
        ...     dataset_name="R1_L100",
        ...     nb_subjects_per_shard=25,
        ...     tasks=["RestingState", "surroundSupp", "DespicableMe"],
        ...     epoch_length=2.0
        ... )
    """
    if verbose:
        print('=' * 70)
        print('  CRL Multi-Task Pretraining Dataset Creation')
        print('=' * 70)

    # Use default tasks if none specified
    if tasks is None:
        tasks = DEFAULT_TASKS

    # Setup loader
    config = SimpleConfig(data_root=database_root, dataset_name=dataset_name)
    loader = SimpleHBNLoader(config)
    all_subjects = loader.get_available_subjects()

    if subjects is None:
        subjects = all_subjects

    # Create output directory
    savepath_root.mkdir(parents=True, exist_ok=True)

    if verbose:
        print(f"  Dataset:        {dataset_name}")
        print(f"  Total subjects: {len(subjects)}")
        print(f"  Tasks:          {', '.join(tasks)}")
        print(f"  Runs:           {runs}")
        print(f"  Epoch length:   {epoch_length}s")
        print(f"  Overlap:        {overlap}s")
        print(f"  Subjects/shard: {nb_subjects_per_shard}")
        print(f"  Output dir:     {savepath_root}")
        print('=' * 70)
        print()

    # ========== Multi-Task Data Collection ==========
    pretraining_data = []
    shard_counter = 0

    for subject_idx, SUBJECT in enumerate(subjects):
        subject_epochs_count = 0

        # Loop through all tasks
        for TASK in tasks:
            for RUN in runs:
                if verbose:
                    print(f"[{subject_idx+1}/{len(subjects)}] {SUBJECT} | {TASK} | Run {RUN}", end=" ")

                # Check if data exists
                if not loader.data_exists(SUBJECT, TASK, run=RUN):
                    if verbose:
                        print("✗ Not found")
                    continue

                try:
                    # Load data
                    data = loader.get_data(SUBJECT, TASK, run=RUN)
                    sfreq = data['raw'].info['sfreq']

                    # Preprocess
                    preprocessed_data = preprocess_data(
                        data,
                        notch_freq=60.0,
                        bandpass_freq=(0.5, 40.0),
                        ref_channels='average',
                        zscore_method='channel_wise'
                    )
                    del data

                    # Segment into epochs
                    epochs = segment_continuous_numpy(
                        preprocessed_data,
                        sfreq,
                        epoch_length,
                        overlap=overlap
                    )
                    del preprocessed_data

                    if verbose:
                        print(f"✓ {epochs.shape[0]} epochs")

                    # Add to dataset with metadata
                    for epoch in epochs:
                        pretraining_data.append({
                            'signal': epoch,
                            'subject': SUBJECT,
                            'task': TASK,
                            'run': RUN
                        })
                        subject_epochs_count += 1

                    del epochs

                except Exception as e:
                    if verbose:
                        print(f"✗ Error: {str(e)[:50]}")
                    continue

        # Save shard every nb_subjects_per_shard subjects
        if (subject_idx + 1) % nb_subjects_per_shard == 0:
            if pretraining_data:
                temp_df = pd.DataFrame(pretraining_data)
                shard_path = savepath_root / f'crl_pretraining_data_shard_{shard_counter}.pkl'

                with open(shard_path, 'wb') as f:
                    pickle.dump(temp_df, f)

                if verbose:
                    print()
                    print(f"{'='*70}")
                    print(f"  ✓ Saved shard {shard_counter}: {len(temp_df)} epochs")
                    print(f"    Path: {shard_path.name}")
                    print(f"{'='*70}")
                    print()

                shard_counter += 1
                pretraining_data = []

    # Save remaining data
    if pretraining_data:
        temp_df = pd.DataFrame(pretraining_data)
        shard_path = savepath_root / f'crl_pretraining_data_shard_{shard_counter}.pkl'

        with open(shard_path, 'wb') as f:
            pickle.dump(temp_df, f)

        if verbose:
            print()
            print(f"{'='*70}")
            print(f"  ✓ Saved final shard {shard_counter}: {len(temp_df)} epochs")
            print(f"    Path: {shard_path.name}")
            print(f"{'='*70}")

    if verbose:
        print()
        print(f"{'='*70}")
        print("  ✓ CRL Dataset Creation Complete!")
        print(f"  Total shards created: {shard_counter + 1}")
        print(f"{'='*70}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create multi-task CRL pretraining shards from HBN EEG database",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create shards with all 6 tasks (default)
  python database_to_crl_pretraining_shards.py

  # Use specific tasks only
  python database_to_crl_pretraining_shards.py \\
      --tasks RestingState surroundSupp DespicableMe

  # Customize shard size and epoch length
  python database_to_crl_pretraining_shards.py \\
      --subjects-per-shard 50 \\
      --epoch-length 2.0 \\
      --overlap 0.5
        """
    )

    parser.add_argument(
        "--dataset-name",
        default="R1_L100",
        help="Dataset name (default: R1_L100)"
    )
    parser.add_argument(
        "--database-root",
        type=Path,
        default=Path("/home/mts/HBN_EEG_v1/database/"),
        help="Database root path"
    )
    parser.add_argument(
        "--savepath-root",
        type=Path,
        default=Path("/home/mts/HBN_EEG_v1/datasets/"),
        help="Output directory for shards"
    )
    parser.add_argument(
        "--subjects-per-shard",
        type=int,
        default=25,
        help="Number of subjects per shard (default: 25)"
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help=f"Tasks to include (default: all 6 tasks). Available: {', '.join(DEFAULT_TASKS)}"
    )
    parser.add_argument(
        "--runs",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Run numbers to process (default: 1 2)"
    )
    parser.add_argument(
        "--epoch-length",
        type=float,
        default=2.0,
        help="Epoch length in seconds (default: 2.0)"
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=0.0,
        help="Overlap between epochs in seconds (default: 0.0)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )

    args = parser.parse_args()

    create_crl_pretraining_shards(
        dataset_name=args.dataset_name,
        database_root=args.database_root,
        savepath_root=args.savepath_root,
        nb_subjects_per_shard=args.subjects_per_shard,
        tasks=args.tasks,
        runs=args.runs,
        epoch_length=args.epoch_length,
        overlap=args.overlap,
        verbose=args.verbose
    )
