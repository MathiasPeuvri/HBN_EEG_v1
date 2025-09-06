"""
Dataset for downstream tasks (classification or regression)
"""
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .. import config


class DownstreamTaskDataset(Dataset):
    """Dataset for downstream tasks (classification or regression)"""
    
    def __init__(self, data_path: str = None, task_type: str = None):
        """
        Initialize downstream task dataset
        
        Args:
            data_path: Path to data pickle file (uses DOWNSTREAM_DATA_PATH if None)
            task_type: 'classification' or 'regression' (uses config default if None)
        """
        if task_type is None:
            task_type = config.TASK_TYPE
        if data_path is None:
            data_path = config.DOWNSTREAM_DATA_PATH
        
        self.task_type = task_type
            
        with open(data_path, 'rb') as f:
            self.df = pickle.load(f)
        
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
        
        # Filter data based on TARGET_EVENTS if specified (skip for psychopathology factors)
        if config.TARGET_EVENTS is not None and not self.is_psychopathology_target:
            mask = self.df[config.TARGET_COLUMN].isin(config.TARGET_EVENTS)
            self.df_filtered = self.df[mask].copy()
        else:
            # Use all data if no events specified or for psychopathology factors
            self.df_filtered = self.df.copy()
            
        # For classification, auto-generate label mapping alphabetically
        if task_type == 'classification':
            unique_values = sorted(self.df_filtered[config.TARGET_COLUMN].unique())
            self.label_map = {value: i for i, value in enumerate(unique_values)}
            print(f"Auto-generated label mapping: {self.label_map}")
        else:
            self.label_map = None
        
        # Extract signals and labels
        self.signals = []
        self.labels = []
        
        for _, row in self.df_filtered.iterrows():
            signal = row['signal']
            # Ensure signal has correct shape
            if signal.shape == (config.NUM_CHANNELS, config.POSTTRAINING_SEQ_LEN):
                self.signals.append(signal.astype(np.float32))
                
                # Extract target value
                if self.is_psychopathology_target:
                    # Get subject ID and lookup psychopathology factor
                    subject_id = row['subject']
                    if subject_id in self.psycho_factor_map:
                        target_value = self.psycho_factor_map[subject_id]
                    else:
                        # Skip subjects not in participants.tsv
                        self.signals.pop()  # Remove the signal we just added
                        continue
                else:
                    # Use column from data as before
                    target_value = row[config.TARGET_COLUMN]
                
                # For classification, map to numeric label
                if task_type == 'classification':
                    target = self.label_map[target_value]
                else:
                    target = target_value
                self.labels.append(target)
        
        self.signals = np.array(self.signals)
        self.labels = np.array(self.labels)
        
        # Print informative loading message
        if task_type == 'classification':
            unique_labels, counts = np.unique(self.labels, return_counts=True)
            label_dist = {f"class_{label}": count for label, count in zip(unique_labels, counts)}
            print(f"Loaded {len(self.signals)} samples --> Classification distribution: {label_dist}")
        else:
            if np.issubdtype(self.labels.dtype, np.number):
                target_description = f"psychopathology factor '{config.TARGET_COLUMN}'" if self.is_psychopathology_target else f"target '{config.TARGET_COLUMN}'"
                print(f"Loaded {len(self.signals)} samples --> {target_description} range: [{np.min(self.labels):.3f}, {np.max(self.labels):.3f}]")
            else:
                print(f"Loaded {len(self.signals)} samples for regression task")
        
    # methods for pytorch dataloader
    def __len__(self):
        return len(self.signals)
    
    def __getitem__(self, idx):
        signal = torch.tensor(self.signals[idx], dtype=torch.float32)
        # Use appropriate tensor type based on task type
        if self.task_type == 'classification':
            label = torch.tensor(self.labels[idx], dtype=torch.long)
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return signal, label