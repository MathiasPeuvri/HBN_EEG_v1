"""
EEG epoching and segmentation functions.

Hybrid approach: Use NumPy for simple/transparent epoching,
MNE only for complex event handling and metadata management.
"""

import mne
import numpy as np
from typing import Optional, List, Tuple, Union


def segment_continuous_numpy(data: np.ndarray, sfreq: float,
                            epoch_length: float,
                            overlap: float = 0.0, verbose: bool = False) -> np.ndarray:
    """
    Segment continuous EEG data into fixed-length epochs using NumPy.
    Args:
        data: EEG data array (channels x time)
        sfreq: Sampling frequency
        epoch_length: Length of each epoch in seconds
        overlap: Fraction of overlap between epochs (0-1)
        
    Returns:
        Epoched data array (n_epochs x channels x samples_per_epoch)
    """
    n_channels, n_timepoints = data.shape
    
    # 1. Calculate epoch parameters
    epoch_samples = int(epoch_length * sfreq)
    step_samples = int(epoch_samples * (1 - overlap))
    if step_samples <= 0:
        raise ValueError(f"Step size must be positive, got {step_samples}")
    
    # 2. Calculate number of epochs we can create
    n_epochs = (n_timepoints - epoch_samples) // step_samples + 1
    if n_epochs <= 0:
        raise ValueError(f"Cannot create epochs: data too short ({n_timepoints} samples) "
                        f"for epoch length {epoch_samples} samples")
    
    # 3. Create epochs by slicing
    epochs = np.zeros((n_epochs, n_channels, epoch_samples))
    actual_epochs = 0
    
    for i in range(n_epochs):
        start = i * step_samples
        end = start + epoch_samples
        
        # Ensure we don't go beyond data bounds
        if end <= n_timepoints:
            epochs[i] = data[:, start:end]
            actual_epochs = i + 1
        else:
            # Skip incomplete epochs at the end
            break
    
    if verbose:
        print(f"NumPy segmentation: {epoch_length}s epochs, {overlap:.0%} overlap")
        print(f"Created {actual_epochs} epochs from {n_timepoints} samples")
        print(f"Samples per epoch: {epoch_samples}, Step: {step_samples}")
    
    return epochs[:actual_epochs]


def segment_by_events_mne(raw: mne.io.Raw,
                         events: np.ndarray,
                         event_ids: dict,
                         tmin: float = -0.2,
                         tmax: float = 0.8, verbose: bool = False) -> mne.Epochs:
    """
    Segment EEG data based on events/triggers using MNE.
    Args:
        raw: MNE Raw object
        events: Event array from MNE (n_events x 3: [sample, duration, event_id])
        event_ids: Dictionary mapping event names to IDs
        tmin: Time before event onset (seconds)
        tmax: Time after event onset (seconds)
        
    Returns:
        MNE Epochs object
    """
    if events is None or len(events) == 0:
        raise ValueError("No events provided for epoching")
    
    if not event_ids:
        raise ValueError("Event IDs dictionary cannot be empty")
    
    # Validate that event_ids match events in the array
    available_ids = set(events[:, 2])
    requested_ids = set(event_ids.values())
    
    if not requested_ids.intersection(available_ids):
        raise ValueError(f"None of the requested event IDs {requested_ids} "
                        f"found in events array {available_ids}")
    
    if verbose:
        print(f"MNE event epoching: {tmin} to {tmax}s")
        print(f"Event IDs: {event_ids}")
        print(f"Total events in array: {len(events)}")
    
    # Filter events to only include those in event_ids
    valid_events = events[np.isin(events[:, 2], list(event_ids.values()))]
    if verbose:
        print(f"Valid events for epoching: {len(valid_events)}")
    
    if len(valid_events) == 0:
        raise ValueError("No valid events found for epoching after filtering")
    
    # Create MNE Epochs object with no baseline correction
    epochs = mne.Epochs(
        raw,
        valid_events,
        event_ids,
        tmin=tmin,
        tmax=tmax,
        baseline=None,  # No baseline correction as requested
        preload=True,
        verbose=False
    )
    
    if verbose:
        print(f"Created {len(epochs)} MNE epochs")
    
    return epochs


def segment_by_events_numpy(data: np.ndarray, sfreq: float, 
                           event_times: np.ndarray,
                           tmin: float = -0.2, 
                           tmax: float = 0.8, verbose: bool = False) -> np.ndarray:
    """
    Segment EEG data based on event times using NumPy
    Args:
        data: EEG data array (channels x time)
        sfreq: Sampling frequency
        event_times: Event times in samples (not seconds!)
        tmin: Time before event onset (seconds)
        tmax: Time after event onset (seconds)
        
    Returns:
        Epoched data array (n_events x channels x samples_per_epoch)
    """
    if len(event_times) == 0:
        raise ValueError("No event times provided for epoching")
    
    n_channels, n_timepoints = data.shape
    
    # 1. Convert time window to samples (match MNE's behavior)
    tmin_samples = int(tmin * sfreq)
    tmax_samples = int(tmax * sfreq)
    epoch_length = tmax_samples - tmin_samples # + 1  # +1 to match MNE's inclusive endpoints
    
    if verbose:
        print(f"NumPy event epoching: {tmin} to {tmax}s")
        print(f"Epoch length: {epoch_length} samples")
        print(f"Number of events: {len(event_times)}")
    
    # 2. Initialize output array
    epochs = np.zeros((len(event_times), n_channels, epoch_length))
    valid_epochs = 0
    
    # 3. Extract epochs around each event
    for i, event_sample in enumerate(event_times):
        # Ensure event_sample is integer
        event_sample = int(event_sample)
        
        # Calculate epoch boundaries
        start = event_sample + tmin_samples
        end = event_sample + tmax_samples #+ 1  # +1 for inclusive end
        
        # Check if epoch is within data bounds
        if 0 <= start and end <= n_timepoints:
            epochs[i] = data[:, start:end]
            valid_epochs += 1
        else:
            # Fill with NaN for out-of-bounds epochs
            epochs[i] = np.nan
            print(f"Warning: Event at sample {event_sample} is out of bounds "
                  f"(epoch would span {start}:{end}, data has {n_timepoints} samples)")
    if verbose:
        print(f"Created {valid_epochs} valid epochs out of {len(event_times)} events")
    
    # Remove epochs that are completely NaN (if any)
    valid_mask = ~np.all(np.isnan(epochs), axis=(1, 2))
    if not np.all(valid_mask):
        print(f"Removing {np.sum(~valid_mask)} invalid epochs")
        epochs = epochs[valid_mask]
    
    return epochs



