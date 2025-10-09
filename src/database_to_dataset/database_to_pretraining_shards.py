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

sys.path.insert(0, '/home/mts/HBN_EEG_Analysis')
from src.preprocessing.epoching import segment_continuous_numpy, segment_by_events_numpy
# from src.loader import SimpleHBNLoader
# from src.loader.config import DatabaseLoaderConfig
from src.preprocessing.filters import preprocess_data

from src.loader.simple_loader import SimpleConfig, SimpleHBNLoader

# TODO: clean this script ()
#       check which variables are used and the default values and potential conflicts with epoching.py
#       see if we can make a function to replace the for loops 
#               (can we do pretrain and posttrain/challenge data in the same function ?)
#       Or maybe it's ok like that (we could even just split the file in 3)


def create_pretraining_shards(
    dataset_name: str = "R1_L100",
    database_root: Path = Path("/home/mts/HBN_EEG_Analysis/database/"),
    savepath_root: Path = Path("/home/mts/HBN_EEG_Analysis/datasets/"),
    subjects: Optional[List[str]] = None,
    nb_subjects_per_shard: int = 25,
    task_name: str = "surroundSupp",
    runs: List[int] = [1, 2],
    epoch_length: float = 2.0,
    overlap: float = 0.0,
    verbose: bool = True
) -> None:
    """
    Create pretraining data shards from EEG database.
    """
    if verbose:
        print('================ Start database to dataset ==========================')
    
    config = SimpleConfig(data_root=database_root)
    config = SimpleConfig(dataset_name=dataset_name)
    loader = SimpleHBNLoader(config)
    all_subjects = loader.get_available_subjects()
    
    if subjects is None:
        subjects = all_subjects
    
    # Create savepath_root directory if it doesn't exist
    savepath_root.mkdir(parents=True, exist_ok=True)
    
    #========================================================
    # Pretraining data // on prends tout et on segmente sans prendre en compte les events 
    #========================================================
    pretraining_data = []
    shard_subjects_counter = 0
    
    for i, SUBJECT in enumerate(subjects):
        for RUN in runs:
            if verbose:
                print(f"{SUBJECT} {task_name} {RUN} Loading data")
        # need a checking that the data exists
            if loader.data_exists(SUBJECT, task_name, run=RUN):
                data = loader.get_data(SUBJECT, task_name, run=RUN)
            else:
                print(f"{SUBJECT} {task_name} {RUN} Data not found -- I think because of the fcking change in the challenge rules there is no more need to limit pretraining to surroundSupp")
                continue

            sfreq = data['raw'].info['sfreq']
            preprocessed_data = preprocess_data(data, notch_freq=60.0, bandpass_freq=(0.5, 40.0),
                                                 ref_channels='average', zscore_method='channel_wise')
            del data
            if verbose:
                print(f"{SUBJECT} {task_name} {RUN} Preprocessed data shape: {preprocessed_data.shape} (channels, timepoints)")

            epochs = segment_continuous_numpy(preprocessed_data, sfreq, epoch_length, overlap=overlap)
            del preprocessed_data
            if verbose:
                print(f" Epochs shape: {epochs.shape} (epochs, channels, timepoints)")

            # Create DataFrame rows for each epoch
            for epoch in epochs:
                pretraining_data.append({
                    'signal': epoch,
                    'subject': SUBJECT,
                    'run': RUN
                })
            del epochs
    
        # Save shard every nb_subjects_per_shard subjects
        if (i + 1) % nb_subjects_per_shard == 0:
            temp_df = pd.DataFrame(pretraining_data)
            shard_path = savepath_root / f'pretraining_data_shard_{shard_subjects_counter}.pkl'
            with open(shard_path, 'wb') as f:
                pickle.dump(temp_df, f)
            print(f"Saved shard {shard_subjects_counter} with {len(temp_df)} epochs")
            shard_subjects_counter += 1
            pretraining_data = []

    # Save remaining data if any
    if pretraining_data:
        temp_df = pd.DataFrame(pretraining_data)
        shard_path = savepath_root / f'pretraining_data_shard_{shard_subjects_counter}.pkl'
        with open(shard_path, 'wb') as f:
            pickle.dump(temp_df, f)
        print(f"Saved final shard {shard_subjects_counter} with {len(temp_df)} epochs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create pretraining data shards from EEG database")
    parser.add_argument("--dataset-name", default="R1_L100", help="Dataset name")
    parser.add_argument("--database-root", type=Path, default=Path("/home/mts/HBN_EEG_Analysis/database/"), help="Database root path")
    parser.add_argument("--savepath-root", type=Path, default=Path("/home/mts/HBN_EEG_Analysis/datasets/"), help="Output directory for shards")
    parser.add_argument("--subjects-per-shard", type=int, default=25, help="Number of subjects per shard")
    parser.add_argument("--task-name", default="surroundSupp", help="Task name to process")
    parser.add_argument("--runs", nargs="+", type=int, default=[1, 2], help="Run numbers to process")
    parser.add_argument("--epoch-length", type=float, default=2.0, help="Epoch length in seconds")
    parser.add_argument("--overlap", type=float, default=0.0, help="Overlap between epochs")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    create_pretraining_shards(
        dataset_name=args.dataset_name,
        database_root=args.database_root,
        savepath_root=args.savepath_root,
        nb_subjects_per_shard=args.subjects_per_shard,
        task_name=args.task_name,
        runs=args.runs,
        epoch_length=args.epoch_length,
        overlap=args.overlap,
        verbose=args.verbose
    )
