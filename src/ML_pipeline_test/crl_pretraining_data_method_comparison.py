"""
Minimal script to compare Reconstruction vs Contrastive Learning pretraining approaches

This script defines a simple CNN model (max 7 layers) and trains it with two different
unsupervised methods:
1. Masked Autoencoder Reconstruction (MAE)
2. Contrastive Representation Learning (CRL)

The goal is to compare outputs and demonstrate head replacement for downstream tasks.
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Import existing components
from . import config
from .data_loader_ml import create_dataloaders
from .contrastive_learning.NTXent_loss import NTXentLoss
from .contrastive_learning.augmentations import (
    amplitude_scaling, time_shift_simple, additive_gaussian_noise
)


# ================================ Model Definition ================================

class SimpleCNNEncoder(nn.Module):
    """
    3-layer CNN encoder

    Architecture (3 layers):
        Conv1D(129 -> 64) -> ReLU -> Conv1D(64 -> 32) -> ReLU -> Conv1D(32 -> 16) -> ReLU
    """
    def __init__(self, in_channels=129, seq_len=200):
        super().__init__()
        self.in_channels = in_channels
        self.seq_len = seq_len

        self.encoder = nn.Sequential(
            # Layer 1: 129 -> 64 channels
            nn.Conv1d(in_channels, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),

            # Layer 2: 64 -> 32 channels
            nn.Conv1d(64, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            # Layer 3: 32 -> 16 channels (latent representation)
            nn.Conv1d(32, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(16)
        )

    def forward(self, x):
        """x: (batch, 129, 200) -> (batch, 16, 50)"""
        return self.encoder(x)


class SimpleDecoder(nn.Module):
    """
    Simple 2-layer decoder for reconstruction

    Architecture (2 layers):
        ConvTranspose1D(16 -> 32) -> ReLU -> ConvTranspose1D(32 -> 129)

    Total reconstruction model: 3 (encoder) + 2 (decoder) = 5 layers
    """
    def __init__(self, out_channels=129):
        super().__init__()
        self.out_channels = out_channels

        self.decoder = nn.Sequential(
            # Layer 1: 16 -> 32 channels, upsample
            nn.ConvTranspose1d(16, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(32),

            # Layer 2: 32 -> 129 channels, upsample
            nn.ConvTranspose1d(32, out_channels, kernel_size=5, stride=2, padding=2, output_padding=1),
        )

    def forward(self, z):
        """z: (batch, 16, 50) -> (batch, 129, 200)"""
        return self.decoder(z)


class SimpleProjector(nn.Module):
    """
    Simple 2-layer projector for contrastive learning

    Architecture (2 layers):
        GlobalAvgPool -> Linear(16 -> 64) -> ReLU -> Linear(64 -> 128)

    Total CRL model: 3 (encoder) + 2 (projector) = 5 layers
    """
    def __init__(self, input_channels=16, output_dim=128):
        super().__init__()

        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Linear(input_channels, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, z):
        """z: (batch, 16, 50) -> (batch, 128)"""
        return self.projector(z)


class SimpleDownstreamHead(nn.Module):
    """
    Simple downstream head for regression/classification

    Architecture (2 layers):
        GlobalAvgPool -> Linear(16 -> 32) -> ReLU -> Dropout -> Linear(32 -> output_dim)
    """
    def __init__(self, input_channels=16, output_dim=1, task_type='regression', dropout=0.3):
        super().__init__()
        self.task_type = task_type

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(input_channels, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, output_dim)
        )

    def forward(self, z):
        """z: (batch, 16, 50) -> (batch, output_dim)"""
        out = self.head(z)
        if self.task_type == 'regression':
            return out.squeeze(-1)  # (batch,)
        else:
            return out  # (batch, output_dim) for classification


# ================================ Training Functions ================================

def apply_augmentations(x):
    """Apply random augmentations for contrastive learning"""
    # Convert to numpy for augmentations
    x_np = x.cpu().numpy()
    batch_size = x_np.shape[0]
    augmented = []

    for i in range(batch_size):
        sample = x_np[i]  # (channels, time)

        # Randomly apply 2-3 augmentations
        # 1. Amplitude scaling (0.7 to 1.3)
        if np.random.rand() > 0.5:
            scale = np.random.uniform(0.7, 1.3)
            sample = amplitude_scaling(sample, scale)

        # 2. Time shift (-5% to +5%)
        if np.random.rand() > 0.5:
            shift = np.random.uniform(-0.05, 0.05)
            sample = time_shift_simple(sample, shift)

        # 3. Gaussian noise (scale 0.05 to 0.15)
        if np.random.rand() > 0.5:
            noise_scale = np.random.uniform(0.05, 0.15)
            sample = additive_gaussian_noise(sample, noise_scale)

        augmented.append(sample)

    return torch.tensor(np.array(augmented), dtype=x.dtype, device=x.device)


def train_contrastive_v1(encoder, projector, train_loader, val_loader, epochs=10,
                     lr=3e-4, device='cuda', save_path=None, use_crl_dataset=True):
    """Train encoder-projector with contrastive learning

    Args:
        use_crl_dataset: If True, expects batches as (view1, view2) pairs.
                        If False, expects single batches and applies augmentations.
    """
    print("\n" + "="*60)
    print("Training with CONTRASTIVE LEARNING (CRL)")
    print("="*60)

    encoder = encoder.to(device)
    projector = projector.to(device)

    # Combine encoder + projector parameters
    params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-6)
    criterion = NTXentLoss(temperature=0.1)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training
        encoder.train()
        projector.train()
        train_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):
            
            # CRL dataset returns (view1, view2) pairs already augmented
            x1, x2 = batch
            x1 = x1.to(device)
            x2 = x2.to(device)
            
            # Forward pass
            z1 = encoder(x1)
            z2 = encoder(x2)
            proj1 = projector(z1)
            proj2 = projector(z2)

            # Contrastive loss
            loss = criterion(proj1, proj2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches
        history['train_loss'].append(avg_train_loss)

        # Validation
        encoder.eval()
        projector.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                x1, x2 = batch
                x1 = x1.to(device)
                x2 = x2.to(device)

                z1 = encoder(x1)
                z2 = encoder(x2)
                proj1 = projector(z1)
                proj2 = projector(z2)

                loss = criterion(proj1, proj2)
                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss and save_path:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'projector_state_dict': projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, save_path)

    print(f"\nBest Val Loss: {best_val_loss:.4f}")
    print("="*60)

    return encoder, history

def train_contrastive_v2(encoder, projector, train_loader, val_loader, epochs=10,
                     lr=3e-4, device='cuda', save_path=None, use_crl_dataset=True):
    """Train encoder-projector with contrastive learning

    Args:
        use_crl_dataset: If True, expects batches as (view1, view2) pairs.
                        If False, expects single batches and applies augmentations.
    """
    print("\n" + "="*60)
    print("Training with CONTRASTIVE LEARNING (CRL)")
    print("="*60)

    encoder = encoder.to(device)
    projector = projector.to(device)

    # Combine encoder + projector parameters
    params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-6)
    criterion = NTXentLoss(temperature=0.1)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training
        encoder.train()
        projector.train()
        train_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):

            from .contrastive_learning.augmentations import create_augmented_pair
            from .contrastive_learning.config import transformation_ranges
            augmentation_ranges = transformation_ranges
            num_augmentations = 2

            x = np.array(batch)
            x1, x2, _, _ = create_augmented_pair(x, augmentation_ranges, num_augmentations)
            x1 = torch.tensor(x1, dtype=torch.float32, device=device)
            x2 = torch.tensor(x2, dtype=torch.float32, device=device)

            # Forward pass
            z1 = encoder(x1)
            z2 = encoder(x2)
            proj1 = projector(z1)
            proj2 = projector(z2)

            # Contrastive loss
            loss = criterion(proj1, proj2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches
        history['train_loss'].append(avg_train_loss)

        # Validation
        encoder.eval()
        projector.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):
                x = np.array(batch)
                x1, x2, _, _ = create_augmented_pair(x, augmentation_ranges, num_augmentations)
                x1 = torch.tensor(x1, dtype=torch.float32, device=device)
                x2 = torch.tensor(x2, dtype=torch.float32, device=device)

                z1 = encoder(x1)
                z2 = encoder(x2)
                proj1 = projector(z1)
                proj2 = projector(z2)

                loss = criterion(proj1, proj2)
                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss and save_path:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'projector_state_dict': projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, save_path)

    print(f"\nBest Val Loss: {best_val_loss:.4f}")
    print("="*60)

    return encoder, history

def train_contrastive_v3(encoder, projector, train_loader, val_loader, epochs=10,
                     lr=3e-4, device='cuda', save_path=None, use_crl_dataset=True):
    """Train encoder-projector with contrastive learning

    Args:
        use_crl_dataset: If True, expects batches as (view1, view2) pairs.
                        If False, expects single batches and applies augmentations.
    """
    print("\n" + "="*60)
    print("Training with CONTRASTIVE LEARNING (CRL)")
    print("="*60)

    encoder = encoder.to(device)
    projector = projector.to(device)

    # Combine encoder + projector parameters
    params = list(encoder.parameters()) + list(projector.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-6)
    criterion = NTXentLoss(temperature=0.1)

    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}

    for epoch in range(epochs):
        # Training
        encoder.train()
        projector.train()
        train_loss = 0.0
        num_batches = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False):

            from .contrastive_learning.augmentations_torch import create_augmented_pair_batch_torch
            from .contrastive_learning.config import transformation_ranges
            augmentation_ranges = transformation_ranges
            num_augmentations = 2

            x = batch.to(device)
            x1, x2, _, _ = create_augmented_pair_batch_torch(x,
                transformation_ranges, num_augmentations=2)

            # Forward pass
            z1 = encoder(x1)
            z2 = encoder(x2)
            proj1 = projector(z1)
            proj2 = projector(z2)

            # Contrastive loss
            loss = criterion(proj1, proj2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()
            num_batches += 1

        avg_train_loss = train_loss / num_batches
        history['train_loss'].append(avg_train_loss)

        # Validation
        encoder.eval()
        projector.eval()
        val_loss = 0.0
        num_val_batches = 0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False):

                x = batch.to(device)
                x1, x2, _, _ = create_augmented_pair_batch_torch(x,
                    transformation_ranges, num_augmentations=2)
                z1 = encoder(x1)
                z2 = encoder(x2)
                proj1 = projector(z1)
                proj2 = projector(z2)

                loss = criterion(proj1, proj2)
                val_loss += loss.item()
                num_val_batches += 1

        avg_val_loss = val_loss / num_val_batches
        history['val_loss'].append(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Save best model
        if avg_val_loss < best_val_loss and save_path:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'encoder_state_dict': encoder.state_dict(),
                'projector_state_dict': projector.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, save_path)

    print(f"\nBest Val Loss: {best_val_loss:.4f}")
    print("="*60)

    return encoder, history
# ================================ Utility Functions ================================

def replace_head(encoder, head_type='regression', output_dim=1, freeze_encoder=True):
    """
    Replace encoder head for downstream tasks

    Args:
        encoder: Pretrained encoder
        head_type: 'regression' or 'classification'
        output_dim: Output dimension (1 for regression, num_classes for classification)
        freeze_encoder: Whether to freeze encoder weights

    Returns:
        Full model with encoder + downstream head
    """
    if freeze_encoder:
        for param in encoder.parameters():
            param.requires_grad = False

    downstream_head = SimpleDownstreamHead(
        input_channels=16,
        output_dim=output_dim,
        task_type=head_type
    )

    class DownstreamModel(nn.Module):
        def __init__(self, encoder, head):
            super().__init__()
            self.encoder = encoder
            self.head = head

        def forward(self, x):
            z = self.encoder(x)
            return self.head(z)

    model = DownstreamModel(encoder, downstream_head)
    print(f"\nCreated downstream model with {head_type} head (encoder frozen: {freeze_encoder})")

    return model


def main():
    parser = argparse.ArgumentParser(description='Compare Reconstruction vs Contrastive Learning')
    parser.add_argument('--method', type=str, choices=['contrastive_v1', 'contrastive_v2', 'contrastive_v3', 'all'],
                       default='both', help='Training method to use')
    parser.add_argument('--epochs', type=int, default=10,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-3,
                       help='Learning rate (will use 3e-4 for contrastive)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to train on')
    parser.add_argument('--save-dir', type=str, default=None,
                       help='Directory to save models (default: config.MODEL_DIR)')
    parser.add_argument('--use-crl-data', action='store_true',
                       help='Use CRL pretraining data shards instead of standard pretraining data')

    args = parser.parse_args()

    # Handle 'all' method
    if args.method == 'all':
        for method in ['contrastive_v1', 'contrastive_v2', 'contrastive_v3']:
            args.method = method
            print(f"\n{'='*80}\n  Running method: {method}\n{'='*80}\n")
            main_single_method(args)
        return

    main_single_method(args)


def main_single_method(args):
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)

    # Setup save directory
    save_dir = Path(args.save_dir) if args.save_dir else config.MODEL_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading data...")
    if args.method == 'contrastive_v1':
        from .datasets_loader_classes.shard_crl_dataset import ContrastiveShardDataset
        #Load contrastive dataset from pkl shards
        train_dataset = ContrastiveShardDataset(
            shard_pattern=str(config.DATA_DIR / "crl_pretraining_data_shard_*_R1t.pkl"),
            train_split=1.0,
            is_train=True,
            seed=42)
        val_dataset = ContrastiveShardDataset(
            shard_pattern=str(config.DATA_DIR / "crl_pretraining_data_shard_*_t_R1.pkl"),
            train_split=0.0,
            is_train=False,
            seed=42)
        train_loader_crl = DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                num_workers=4,
                pin_memory=args.device == 'cuda')
        val_loader_crl = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=args.device == 'cuda')
    elif args.method in ['contrastive_v2', 'contrastive_v3']:
        from .datasets_loader_classes.shard_crl_dataset import SimplePretrainShardDataset

        train_dataset = SimplePretrainShardDataset(
            shard_pattern=str(config.DATA_DIR / "crl_pretraining_data_shard_*_R1t.pkl"),
            seed=42)
        val_dataset = SimplePretrainShardDataset(
            shard_pattern=str(config.DATA_DIR / "crl_pretraining_data_shard_*_t_R1.pkl"),
            seed=42)

        train_loader_crl = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=args.device == 'cuda'
        )
        val_loader_crl = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=4,
            pin_memory=args.device == 'cuda'
        )
    else:
        print('error, should try one of the available contrastive method')

    # Train with contrastive learning
    if args.method == 'contrastive_v1':
        encoder_crl = SimpleCNNEncoder(in_channels=129, seq_len=200)
        projector = SimpleProjector(input_channels=16, output_dim=128)
        import time
        t0_contrastive_training_v1 = time.time()
        encoder_crl, hist_crl = train_contrastive_v1(
            encoder_crl, projector, train_loader_crl, val_loader_crl,
            epochs=args.epochs, lr=3e-4, device=args.device,
            save_path=save_dir / "simple_contrastive_best.pth",
            use_crl_dataset=args.use_crl_data
        )
        t1_contrastive_training_v1 = time.time()
        print('Contrastive v1 : contrastive signals created in ContrastiveShardDataset , before pythorch DataLoader')
        print(f"Contrastive learning training time: {t1_contrastive_training_v1 - t0_contrastive_training_v1:.2f} seconds")
        print(f"\nContrastive learning training completed!")
        print(f"Final train loss: {hist_crl['train_loss'][-1]:.4f}")
        print(f"Final val loss: {hist_crl['val_loss'][-1]:.4f}")
        print(f"Model saved to: {save_dir / 'simple_contrastive_best.pth'}")
    elif args.method == 'contrastive_v2':
        encoder_crl = SimpleCNNEncoder(in_channels=129, seq_len=200)
        projector = SimpleProjector(input_channels=16, output_dim=128)
        import time
        t0_contrastive_training_v2 = time.time()
        encoder_crl, hist_crl = train_contrastive_v2(
            encoder_crl, projector, train_loader_crl, val_loader_crl,
            epochs=args.epochs, lr=3e-4, device=args.device,
            save_path=save_dir / "simple_contrastive_best.pth",
            use_crl_dataset=args.use_crl_data
        )
        t1_contrastive_training_v2 = time.time()
        print('Contrastive v2 : Contrastive views created during training but with numpy arrays')

        print(f"Contrastive learning training time: {t1_contrastive_training_v2 - t0_contrastive_training_v2:.2f} seconds")
        print(f"\nContrastive learning training completed!")
        print(f"Final train loss: {hist_crl['train_loss'][-1]:.4f}")
        print(f"Final val loss: {hist_crl['val_loss'][-1]:.4f}")
        print(f"Model saved to: {save_dir / 'simple_contrastive_best.pth'}")
    elif args.method == 'contrastive_v3':
        encoder_crl = SimpleCNNEncoder(in_channels=129, seq_len=200)
        projector = SimpleProjector(input_channels=16, output_dim=128)
        import time
        t0_contrastive_training_v3 = time.time()
        encoder_crl, hist_crl = train_contrastive_v3(
            encoder_crl, projector, train_loader_crl, val_loader_crl,
            epochs=args.epochs, lr=3e-4, device=args.device,
            save_path=save_dir / "simple_contrastive_best.pth",
            use_crl_dataset=args.use_crl_data
        )
        t1_contrastive_training_v3 = time.time()
        print('Contrastive v3 : Full GPU contrastive signals creation // Check if use batch or signal version')

        print(f"Contrastive learning training time: {t1_contrastive_training_v3 - t0_contrastive_training_v3:.2f} seconds")
        print(f"\nContrastive learning training completed!")
        print(f"Final train loss: {hist_crl['train_loss'][-1]:.4f}")
        print(f"Final val loss: {hist_crl['val_loss'][-1]:.4f}")
        print(f"Model saved to: {save_dir / 'simple_contrastive_best.pth'}")

    # print(f'Time comparison : ')
    # print(f"Contrastive v1: {t1_contrastive_training_v1 - t0_contrastive_training_v1:.2f} seconds")
    # print(f"Contrastive v2: {t1_contrastive_training_v2 - t0_contrastive_training_v2:.2f} seconds")
    # print(f"Contrastive v3: {t1_contrastive_training_v3 - t0_contrastive_training_v3:.2f} seconds")


if __name__ == "__main__":
    main()
