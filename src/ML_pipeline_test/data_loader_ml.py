"""
Data loading and preprocessing for EEG ML pipeline
"""
import glob
import torch
from torch.utils.data import DataLoader, random_split
from typing import Tuple, Optional
from . import config

# Import dataset classes from new modules
from .datasets_loader_classes.ssl_dataset import SSL_Dataset
from .datasets_loader_classes.shard_pretraining_dataset import SequentialShardDataset
from .datasets_loader_classes.downstream_dataset import DownstreamTaskDataset
from .datasets_loader_classes.shard_downstream_dataset import SequentialShardDownstreamDataset

NUM_WORKERS = 0

def create_dataloaders(dataset_type: str = 'pretraining',
                       batch_size: Optional[int] = None,
                       train_split: float = config.TRAIN_SPLIT,
                       data_format: str = "v1") -> Tuple[DataLoader, DataLoader]:
    """
    Create train and validation dataloaders with sharding support

    Args:
        dataset_type: 'pretraining', 'classification', or 'regression'
        batch_size: Batch size for dataloaders
        train_split: Proportion of data for training
        data_format: chall 1 standard or inwindow_rtidx_augmentation approach

    Returns:
        train_loader, val_loader
    """
    if dataset_type == 'pretraining':
        # Check if we have multiple shards or single file
        pattern = config.PRETRAINING_DATA_PATTERN
        
        if '*' in pattern and len(glob.glob(pattern)) > 1:
            # Multiple shards - use SequentialShardDataset
            train_dataset = SequentialShardDataset(
                pattern, 
                train_split=train_split,
                is_train=True,
                seed=config.RANDOM_SEED
            )
            val_dataset = SequentialShardDataset(
                pattern,
                train_split=train_split, 
                is_train=False,
                seed=config.RANDOM_SEED
            )
            
            if batch_size is None:
                batch_size = config.AE_BATCH_SIZE
            
            # IterableDataset requires different DataLoader setup
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=NUM_WORKERS,  
                pin_memory=config.DEVICE.type == 'cuda'
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=NUM_WORKERS,
                pin_memory=config.DEVICE.type == 'cuda'
            )
            
            print(f"Created sharded dataloaders with pattern: {pattern}")
            return train_loader, val_loader
        else:
            # Single file - use existing SSL_Dataset
            if '*' in pattern:
                # Pattern exists but only one file matches
                data_path = glob.glob(pattern)[0] if glob.glob(pattern) else pattern
            else:
                data_path = pattern
            dataset = SSL_Dataset(data_path)
            if batch_size is None:
                batch_size = config.AE_BATCH_SIZE
    elif dataset_type == 'classification':
        # Check if we have multiple shards or single file
        pattern = config.DOWNSTREAM_DATA_PATTERN

        if '*' in pattern and len(glob.glob(pattern)) > 1:
            # Multiple shards - use SequentialShardDownstreamDataset
            train_dataset = SequentialShardDownstreamDataset(
                pattern,
                task_type='classification',
                train_split=train_split,
                is_train=True,
                seed=config.RANDOM_SEED,
                data_format=data_format
            )
            val_dataset = SequentialShardDownstreamDataset(
                pattern,
                task_type='classification',
                train_split=train_split,
                is_train=False,
                seed=config.RANDOM_SEED,
                data_format=data_format
            )
            
            if batch_size is None:
                batch_size = config.CLS_BATCH_SIZE
            
            # IterableDataset requires different DataLoader setup
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=NUM_WORKERS,  # Start with 0, can optimize later
                pin_memory=config.DEVICE.type == 'cuda'
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=NUM_WORKERS,
                pin_memory=config.DEVICE.type == 'cuda'
            )
            
            print(f"Created sharded downstream dataloaders (classification) with pattern: {pattern}")
            return train_loader, val_loader
        else:
            # Single file - use existing DownstreamTaskDataset
            if '*' in pattern:
                # Pattern exists but only one file matches
                data_path = glob.glob(pattern)[0] if glob.glob(pattern) else pattern
            else:
                data_path = pattern
            dataset = DownstreamTaskDataset(data_path=data_path, task_type='classification')
            if batch_size is None:
                batch_size = config.CLS_BATCH_SIZE
    elif dataset_type == 'regression':
        # Check if we have multiple shards or single file
        pattern = config.DOWNSTREAM_DATA_PATTERN

        if '*' in pattern and len(glob.glob(pattern)) > 1:
            # Multiple shards - use SequentialShardDownstreamDataset
            train_dataset = SequentialShardDownstreamDataset(
                pattern,
                task_type='regression',
                train_split=train_split,
                is_train=True,
                seed=config.RANDOM_SEED,
                data_format=data_format
            )
            val_dataset = SequentialShardDownstreamDataset(
                pattern,
                task_type='regression',
                train_split=train_split,
                is_train=False,
                seed=config.RANDOM_SEED,
                data_format=data_format
            )
            
            if batch_size is None:
                batch_size = config.CLS_BATCH_SIZE
            
            # IterableDataset requires different DataLoader setup
            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                num_workers=NUM_WORKERS, 
                pin_memory=config.DEVICE.type == 'cuda'
            )
            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                num_workers=NUM_WORKERS,
                pin_memory=config.DEVICE.type == 'cuda'
            )
            
            print(f"Created sharded downstream dataloaders (regression) with pattern: {pattern}")
            return train_loader, val_loader
        else:
            # Single file - use existing DownstreamTaskDataset
            if '*' in pattern:
                # Pattern exists but only one file matches
                data_path = glob.glob(pattern)[0] if glob.glob(pattern) else pattern
            else:
                data_path = pattern
            dataset = DownstreamTaskDataset(data_path=data_path, task_type='regression')
            if batch_size is None:
                batch_size = config.CLS_BATCH_SIZE
    else:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    
    # For non-sharded datasets, use the existing split logic
    # Calculate split sizes
    train_size = int(train_split * len(dataset))
    val_size = len(dataset) - train_size
    
    # Split dataset
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.RANDOM_SEED))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if config.DEVICE.type == 'cuda' else False
    )
    
    print(f"Created dataloaders: train={len(train_dataset)}, val={len(val_dataset)}")
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Test data loading
    print("Testing SSL Dataset...")
    ssl_dataset = SSL_Dataset()
    print(f"SSL dataset size: {len(ssl_dataset)}")
    sample = ssl_dataset[0]
    print(f"Sample shape: {sample.shape}")
    
    print("\nTesting Downstream Task Dataset...")
    cls_dataset = DownstreamTaskDataset()
    print(f"Downstream dataset size: {len(cls_dataset)}")
    signal, label = cls_dataset[0]
    print(f"Signal shape: {signal.shape}, Label: {label}")
    
    print("\nTesting DataLoaders...")
    train_loader, val_loader = create_dataloaders('pretraining')
    batch = next(iter(train_loader))
    print(f"Pretraining batch shape: {batch.shape}")
    
    train_loader, val_loader = create_dataloaders('classification')
    signals, labels = next(iter(train_loader))
    print(f"Classification batch shapes: signals={signals.shape}, labels={labels.shape}")