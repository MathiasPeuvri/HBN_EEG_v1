import pickle
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import warnings
mne.set_log_level('ERROR')
warnings.filterwarnings("ignore", category=RuntimeWarning) 

sys.path.insert(0, '/home/mts/HBN_EEG_v1')
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


print('================ Start database to dataset ==========================')
SUBJECTS = ["NDARAC904DMU", "NDARZH761YA7", "NDARCE721YB5"]
TASK_PRET = "surroundSupp" #2 runs
RUNS_PRET = [1, 2]
EPOCH_LENGTH = 2.0  # seconds per epoch
OVERLAP = 0.0

VERBOSE = True

# config = DatabaseLoaderConfig(data_root=Path("/home/mts/EFG2025_HBN-EEG_databse/data"))
# loader = SimpleHBNLoader(config)
config = SimpleConfig(data_root=Path("/home/mts/HBN_EEG_v1/database/"))
loader = SimpleHBNLoader(config)
all_subjects = loader.get_available_subjects()
SUBJECTS = all_subjects[:80] # test with 25 subjects at a time for a 'shard' ? // pkl quasi 2 go et ram quasi saturée ... 
                            # consider that here the memory issue happen when trying to store epochs from 2 files per subject
                            # (epochs are stored in a list and then converted to a dataframe)
                            # sharding for full release database is working, but for now let's keep only the 'first shard'


#========================================================
# Pretraining data // on prends tout et on segmente sans prendre en compte les events 
#========================================================
pretraining_data = []
shard_subjects_counter = 0
nb_suj_per_shard = 25
for i, SUBJECT in enumerate(SUBJECTS):
    for RUN in RUNS_PRET:
        if VERBOSE:
            print(f"{SUBJECT} {TASK_PRET} {RUN} Loading data")
        # need a checking that the data exists
        if loader.data_exists(SUBJECT, TASK_PRET, run=RUN):
            data = loader.get_data(SUBJECT, TASK_PRET, run=RUN)
        else:
            print(f"{SUBJECT} {TASK_PRET} {RUN} Data not found")
            continue

        sfreq = data['raw'].info['sfreq']
        preprocessed_data = preprocess_data(data, notch_freq=60.0, bandpass_freq=(0.5, 40.0),
                                             ref_channels='average', zscore_method='channel_wise')
        del data
        if VERBOSE:
            print(f"{SUBJECT} {TASK_PRET} {RUN} Preprocessed data shape: {preprocessed_data.shape} (channels, timepoints)")

        epochs = segment_continuous_numpy(preprocessed_data, sfreq, EPOCH_LENGTH, overlap=OVERLAP)
        del preprocessed_data
        if VERBOSE:
            print(f" Epochs shape: {epochs.shape} (epochs, channels, timepoints)")

        # Create DataFrame rows for each epoch
        for epoch in epochs:
            pretraining_data.append({
                'signal': epoch,
                'subject': SUBJECT,
                'run': RUN
            })
        del epochs
    
    # Save shard every 5 subjects
    if (i + 1) % nb_suj_per_shard == 0:
        temp_df = pd.DataFrame(pretraining_data)
        with open(f'/home/mts/HBN_EEG_v1/datasets/pretraining_data_shard_{shard_subjects_counter}.pkl', 'wb') as f:
            pickle.dump(temp_df, f)
        print(f"Saved shard {shard_subjects_counter} with {len(temp_df)} epochs")
        shard_subjects_counter += 1
        pretraining_data = []

# Save remaining data if any
if pretraining_data:
    temp_df = pd.DataFrame(pretraining_data)
    with open(f'/home/mts/HBN_EEG_v1/datasets/pretraining_data_shard_{shard_subjects_counter}.pkl', 'wb') as f:
        pickle.dump(temp_df, f)
    print(f"Saved final shard {shard_subjects_counter} with {len(temp_df)} epochs")

#========================================================
# Post-training data
#========================================================
TASK_POST = "contrastChangeDetection" #2 runs
RUNS_POST = [1, 2, 3]
TMIN, TMAX = -1.5, 0.5  # Event epoch window
TARGET_EVENTS = ['left_target', 'right_target']
# maybe should change the target events to the moment the user press the button and look at the feedback (smiley or not target) for result correct or not ?
TARGET_EVENTS = ['right_target', 'right_buttonPress', 'left_target', 'left_buttonPress']

def process_events_by_type(raw, events_df, verbose=False):
    """
    Process events and return a DataFrame where each row is an epoch.
    
    Args:
        raw: MNE Raw object
        events_df: Events DataFrame with columns [onset, duration, value, event_code, feedback]
        sfreq: Sampling frequency
        
    Returns:
        pd.DataFrame: DataFrame with columns [onset, duration, value, event_code, feedback, signal]
                     where each row represents one epoch
    """
    if VERBOSE:
        print("=== Event Processing ===")
        print(f"Event types found: {events_df['value'].unique()}")
    
    all_epochs = []
    raw_data = raw.get_data()
    sfreq = raw.info['sfreq']
    
    # Process each target event type
    for event_type in TARGET_EVENTS:
        if VERBOSE:
            print(f"\n--- Processing {event_type} ---")
        
        # Filter events for this type
        events_this_type = events_df[events_df['value'] == event_type].copy()
        if VERBOSE:
            print(f"Found {len(events_this_type)} {event_type} events")
        
        if len(events_this_type) == 0:
            if VERBOSE:
                print(f"No {event_type} events found, skipping")
            continue
        
        # Create epochs for this event type
        try:
            event_times_samples = (events_this_type['onset'].values * sfreq).astype(int)
            numpy_epochs = segment_by_events_numpy(
                raw_data, sfreq, event_times_samples, tmin=TMIN, tmax=TMAX
            )
            if VERBOSE:
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
            if VERBOSE:
                print(f"NumPy epoching failed for {event_type}: {e}")
            continue
    
    # Create final DataFrame
    epochs_df = pd.DataFrame(all_epochs)
    if VERBOSE:
        print(f"Created epochs DataFrame with shape: {epochs_df.shape}")
        print(f"Signal column contains arrays of shape: {epochs_df['signal'].iloc[0].shape if len(epochs_df) > 0 else 'N/A'}")
    
    return epochs_df


posttraining_data = pd.DataFrame()
for SUBJECT in SUBJECTS:
    for RUN in RUNS_POST:
        data = loader.get_data(SUBJECT, TASK_POST, run=RUN)
        sfreq = data['raw'].info['sfreq']

        preprocessed_data = preprocess_data(data, notch_freq=60.0, bandpass_freq=(0.5, 40.0),
                                             ref_channels='average', zscore_method='channel_wise')
        if VERBOSE:
            print(f" Preprocessed data shape: {preprocessed_data.shape} (channels, timepoints)")

        epochs_df = process_events_by_type(data['raw'], data['events'], verbose=VERBOSE)
        epochs_df['subject'] = SUBJECT
        epochs_df['run'] = RUN
        if VERBOSE:
            print(f" Epochs DataFrame shape: {epochs_df.shape}")
        if len(epochs_df) > 0:
            if VERBOSE:
                print(f" Sample epoch signal shape: {epochs_df['signal'].iloc[0].shape}")
                print(f" Event types in epochs: {epochs_df['value'].unique()}")

        posttraining_data = pd.concat([posttraining_data, epochs_df], ignore_index=True)

print(f"Posttraining data shape: {posttraining_data.shape} ({posttraining_data.value.unique()}, with signal shape: {posttraining_data['signal'].iloc[0].shape})")

print(posttraining_data)

#========================================================
# Challenge task
#========================================================
# Set challenge task parameters
TMIN, TMAX = -2.0, 0.0  # Pre-trial epochs
TARGET_EVENTS = ['right_target', 'left_target']  # Only targets

def extract_challenge1_data(epochs_df, events_df, verbose=False):
    """
    Simplified version using process_events_by_type output.
    
    Args:
        epochs_df: Output from process_events_by_type with all events
        events_df: Original events DataFrame for response matching
        
    Returns:
        pd.DataFrame: Challenge data with response metrics
    """
    # Filter to only target events (pre-trial epochs)
    target_epochs = epochs_df[epochs_df['value'].isin(['right_target', 'left_target'])].copy()
    
    if VERBOSE:
        print(f"Found {len(target_epochs)} target epochs")
    
    # Add response metrics for each target
    for idx, target_row in target_epochs.iterrows():
        target_time = target_row['onset']
        target_type = target_row['value']
        target_side = 'right' if target_type == 'right_target' else 'left'
        
        # Find next button press after this target
        button_events = events_df[
            (events_df['value'].isin(['right_buttonPress', 'left_buttonPress'])) &
            (events_df['onset'] > target_time)].sort_values('onset')
        
        # Calculate response metrics
        if len(button_events) > 0:
            response_event = button_events.iloc[0]
            response_time = response_event['onset'] - target_time
            response_side = 'right' if response_event['value'] == 'right_buttonPress' else 'left'
            hit_accuracy = 'correct' if response_event['feedback'] == 'smiley_face' else 'incorrect'
        else:
            response_time = np.nan
            response_side = 'no_response'
            hit_accuracy = 'no_response'
        
        # Add response metrics to this row
        target_epochs.at[idx, 'target_side'] = target_side
        target_epochs.at[idx, 'response_time'] = response_time
        target_epochs.at[idx, 'response_side'] = response_side
        target_epochs.at[idx, 'hit_accuracy'] = hit_accuracy
    
    return target_epochs




challenge_1_data = pd.DataFrame()
print("\n=== CORRECTED CHALLENGE DATA EXTRACTION V2 ===")
for SUBJECT in SUBJECTS:
    for RUN in RUNS_POST:
        data = loader.get_data(SUBJECT, TASK_POST, run=RUN)
        sfreq = data['raw'].info['sfreq']
        preprocessed_data = preprocess_data(data, notch_freq=60.0, bandpass_freq=(0.5, 40.0),
                                        ref_channels='average', zscore_method='channel_wise')
        
        # Create preprocessed raw object
        raw_preprocessed = data['raw'].copy()
        raw_preprocessed._data = preprocessed_data
        
        # Get epochs DataFrame using process_events_by_type
        epochs_df = process_events_by_type(raw_preprocessed, data['events'], verbose=VERBOSE)
        
        epochs_df = extract_challenge1_data(epochs_df, data['events'], verbose=VERBOSE)
        
        # Add subject and run info
        epochs_df['subject'] = SUBJECT
        epochs_df['run'] = RUN
        
        challenge_1_data = pd.concat([challenge_1_data, epochs_df], ignore_index=True)

print(f"\nCorrected posttraining data shape: {challenge_1_data.shape}")
print(f"Signal shape per epoch: {challenge_1_data['signal'].iloc[0].shape}")
print(f"Hit accuracy distribution: {challenge_1_data['hit_accuracy'].value_counts()}")
print(f"Mean response time: {challenge_1_data['response_time'].mean():.3f}s")

print(challenge_1_data.head())



#========================================================
# save the data
#========================================================
with open('/home/mts/HBN_EEG_v1/datasets/pretraining_data.pkl', 'wb') as f:
        pickle.dump(pretraining_data, f)

with open('/home/mts/HBN_EEG_v1/datasets/posttraining_data.pkl', 'wb') as f:
    pickle.dump(posttraining_data, f)

# Save corrected data
with open('/home/mts/HBN_EEG_v1/datasets/challenge_1_data.pkl', 'wb') as f:
    pickle.dump(challenge_1_data, f)
