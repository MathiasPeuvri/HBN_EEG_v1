import pickle
import sys
import argparse
from pathlib import Path
from typing import Optional, List
import pandas as pd
import mne
import warnings
mne.set_log_level('ERROR')
warnings.filterwarnings("ignore", category=RuntimeWarning) 
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

sys.path.insert(0, str(PROJECT_ROOT))
from src.preprocessing.epoching import segment_by_events_numpy
from src.preprocessing.filters import preprocess_data
from src.loader.simple_loader import SimpleConfig, SimpleHBNLoader


def process_events_by_type(raw, events_df, verbose=False):
    """
    Process events and return a DataFrame where each row is an epoch.
    
    Args:
        raw: MNE Raw object
        events_df: Events DataFrame with columns [onset, duration, value, event_code, feedback]
        
    Returns:
        pd.DataFrame: DataFrame with columns [onset, duration, value, event_code, feedback, signal]
                     where each row represents one epoch
    """
    if verbose:
        print("=== Event Processing ===")
        print(f"Event types found: {events_df['value'].unique()}")
    
    all_epochs = []
    raw_data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    # Process each target event type
    for event_type in TARGET_EVENTS:
        if verbose:
            print(f"\n--- Processing {event_type} ---")
        
        # Filter events for this type
        events_this_type = events_df[events_df['value'] == event_type].copy()
        if verbose:
            print(f"Found {len(events_this_type)} {event_type} events")
        
        if len(events_this_type) == 0:
            if verbose:
                print(f"No {event_type} events found, skipping")
            continue
        
        # Create epochs for this event type
        try:
            event_times_samples = (events_this_type['onset'].values * sfreq).astype(int)
            numpy_epochs = segment_by_events_numpy(
                raw_data, sfreq, event_times_samples, tmin=TMIN, tmax=TMAX
            )
            if verbose:
                print(f"NumPy epochs shape: {numpy_epochs.shape}")
            
            # Create DataFrame rows for each epoch
            for i, epoch_signal in enumerate(numpy_epochs):
                epoch_row = {
                    'onset': events_this_type.iloc[i]['onset'],
                    'duration': events_this_type.iloc[i]['duration'],
                    'value': events_this_type.iloc[i]['value'],
                    'event_code': events_this_type.iloc[i]['event_code'],
                    'feedback': events_this_type.iloc[i]['feedback'],
                    'signal': epoch_signal  # Shape: (channels, timepoints)
                }
                all_epochs.append(epoch_row)
                
        except Exception as e:
            if verbose:
                print(f"NumPy epoching failed for {event_type}: {e}")
            continue
    
    # Create final DataFrame
    epochs_df = pd.DataFrame(all_epochs)
    if verbose and len(epochs_df) > 0:
        print(f"Created epochs DataFrame with shape: {epochs_df.shape}")
        print(f"Signal column contains arrays of shape: {epochs_df['signal'].iloc[0].shape}")
    
    return epochs_df


def create_posttraining_shards(
    dataset_name: str = "R1_L100",
    database_root: Path = PROJECT_ROOT / "database",
    savepath_root: Path = PROJECT_ROOT / "datasets",
    subjects: Optional[List[str]] = None,
    nb_subjects_per_shard: int = 25,
    task_name: str = "contrastChangeDetection",
    runs: List[int] = [1, 2, 3],
    tmin: float = -1.5,
    tmax: float = 0.5,
    target_events: List[str] = ['right_target', 'right_buttonPress', 'left_target', 'left_buttonPress'],
    verbose: bool = True
) -> None:
    """
    Create posttraining data shards from EEG database.
    """
    # Make target_events available to process_events_by_type function
    global TARGET_EVENTS, TMIN, TMAX
    TARGET_EVENTS = target_events
    TMIN, TMAX = tmin, tmax
    
    if verbose:
        print('================ Start Posttraining Sharding ========================')
    
    # Initialize loader
    config = SimpleConfig(data_root=database_root)
    config = SimpleConfig(dataset_name=dataset_name)
    loader = SimpleHBNLoader(config)
    all_subjects = loader.get_available_subjects()
    
    if subjects is None:
        subjects_list = all_subjects
    else:
        subjects_list = subjects
    
    # Create savepath_root directory if it doesn't exist
    savepath_root.mkdir(parents=True, exist_ok=True)
    
    # Main processing loop with sharding
    posttraining_data = []
    shard_counter = 0
    subjects_processed = 0
    
    if verbose:
        print(f"Starting posttraining data processing for {len(subjects_list)} subjects")
        print(f"Will create shards of {nb_subjects_per_shard} subjects each")

    for i, SUBJECT in enumerate(subjects_list):
        subject_epochs = []
        
        for RUN in runs:
            if verbose:
                print(f"\n{SUBJECT} {task_name} Run {RUN}: Loading data")
            
            # Check if data exists
            if not loader.data_exists(SUBJECT, task_name, run=RUN):
                print(f"{SUBJECT} {task_name} Run {RUN}: Data not found")
                continue
        
            try:
                # Load and preprocess data
                data = loader.get_data(SUBJECT, task_name, run=RUN)
                sfreq = data['raw'].info['sfreq']
                
                preprocessed_data = preprocess_data(
                    data, 
                    notch_freq=60.0, 
                    bandpass_freq=(0.5, 40.0),
                    ref_channels='average', 
                    zscore_method='channel_wise'
                )
                
                if verbose:
                    print(f"Preprocessed data shape: {preprocessed_data.shape} (channels, timepoints)")
            
                # Create preprocessed raw object for event processing
                raw_preprocessed = data['raw'].copy()
                raw_preprocessed._data = preprocessed_data
                
                # Process events
                epochs_df = process_events_by_type(raw_preprocessed, data['events'], verbose=False)
                
                # Add subject and run info
                epochs_df['subject'] = SUBJECT
                epochs_df['run'] = RUN
                
                if verbose:
                    print(f"Created {len(epochs_df)} epochs for this run")
                
                subject_epochs.append(epochs_df)
                
                # Clean up memory
                del data, preprocessed_data, raw_preprocessed, epochs_df
                
            except Exception as e:
                print(f"Error processing {SUBJECT} {task_name} Run {RUN}: {e}")
                continue
    
        # Combine all runs for this subject
        if subject_epochs:
            subject_df = pd.concat(subject_epochs, ignore_index=True)
            posttraining_data.append(subject_df)
            subjects_processed += 1
            
            if verbose:
                print(f"Total epochs for {SUBJECT}: {len(subject_df)}")
            
            del subject_epochs, subject_df
        
        # Save shard every nb_subjects_per_shard subjects
        if (i + 1) % nb_subjects_per_shard == 0 or (i + 1) == len(subjects_list):
            if posttraining_data:
                # Combine all subjects in this shard
                shard_df = pd.concat(posttraining_data, ignore_index=True)
                
                # Save shard
                shard_path = savepath_root / f'posttraining_data_shard_{shard_counter}.pkl'
                with open(shard_path, 'wb') as f:
                    pickle.dump(shard_df, f)
                
                print(f"\n=== Saved shard {shard_counter} ===")
                print(f"Subjects in shard: {subjects_processed}")
                print(f"Total epochs: {len(shard_df)}")
                print(f"File: {shard_path}")
                
                # Reset for next shard
                shard_counter += 1
                subjects_processed = 0
                posttraining_data = []
                del shard_df

    if verbose:
        print(f"\n================ Posttraining Sharding Complete ========================")
        print(f"Total shards created: {shard_counter}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create posttraining data shards from EEG database")
    parser.add_argument("--dataset-name", default="R1_L100", help="Dataset name")
    parser.add_argument("--database-root", type=Path, default=PROJECT_ROOT / "database", help="Database root path")
    parser.add_argument("--savepath-root", type=Path, default=PROJECT_ROOT / "datasets", help="Output directory for shards")
    parser.add_argument("--subjects-per-shard", type=int, default=25, help="Number of subjects per shard")
    parser.add_argument("--task-name", default="contrastChangeDetection", help="Task name to process")
    parser.add_argument("--runs", nargs="+", type=int, default=[1, 2, 3], help="Run numbers to process")
    parser.add_argument("--tmin", type=float, default=-1.5, help="Start time for event epochs (seconds)")
    parser.add_argument("--tmax", type=float, default=0.5, help="End time for event epochs (seconds)")
    parser.add_argument("--target-events", nargs="+", default=['right_target', 'right_buttonPress', 'left_target', 'left_buttonPress'], help="Event types to extract")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    
    args = parser.parse_args()
    
    create_posttraining_shards(
        dataset_name=args.dataset_name,
        database_root=args.database_root,
        savepath_root=args.savepath_root,
        nb_subjects_per_shard=args.subjects_per_shard,
        task_name=args.task_name,
        runs=args.runs,
        tmin=args.tmin,
        tmax=args.tmax,
        target_events=args.target_events,
        verbose=args.verbose
    )