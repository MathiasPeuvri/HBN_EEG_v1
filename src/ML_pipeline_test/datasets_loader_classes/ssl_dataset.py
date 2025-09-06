"""
Dataset for Self-supervised pretraining (masked autoencoding)
"""
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from .. import config


class SSL_Dataset(Dataset):
    """Dataset for Self supervised pretraining (with masked autoencoding) -> only signals"""
    
    def __init__(self, data_path: str = None):
        """
        Initialize EEG dataset for pretraining
        
        Args:
            data_path: Path to pretraining data pickle file
        """
        if data_path is None:
            data_path = config.PRETRAINING_DATA_PATH
            
        with open(data_path, 'rb') as f:
            pretraining_df = pickle.load(f)
        
        if isinstance(pretraining_df, pd.DataFrame):
            # New DataFrame format
            signals = [row['signal'] for _, row in pretraining_df.iterrows()]
            self.data = np.array(signals, dtype=np.float32)
            # Store metadata for potential future use
            self.subjects = pretraining_df['subject'].values
            self.runs = pretraining_df['run'].values
        else:
            # Legacy numpy array format (backward compatibility)
            self.data = pretraining_df.astype(np.float32)
            self.subjects = None
            self.runs = None
        
        # Data shape validation: (epochs, channels, timepoints)
        assert self.data.shape[1] == config.NUM_CHANNELS
        assert self.data.shape[2] == config.PRETRAINING_SEQ_LEN
        
    # methods for pytorch dataloader
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return as (channels, timepoints) for 1D convolution
        sample = torch.tensor(self.data[idx], dtype=torch.float32)
        return sample