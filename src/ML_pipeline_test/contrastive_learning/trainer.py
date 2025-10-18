"""
Training loop for Contrastive Representation Learning

Implements the pretraining strategy from Mohsenvand et al. (2020):
- Learning rate warmup + cosine annealing
- Gradient clipping for LSTM stability
- Validation monitoring and checkpointing
- Optional early stopping
"""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .loss import NTXentLoss
from .config import (
    CRL_CONFIG, EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    WARMUP_EPOCHS, GRAD_CLIP, MIN_LR, CHECKPOINT_DIR, NUM_WORKERS)


def pretrain_contrastive(
    model: nn.Module,
    train_dataset,
    val_dataset=None,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LEARNING_RATE,
    weight_decay: float = WEIGHT_DECAY,
    device: str = 'cuda',
    warmup_epochs: int = WARMUP_EPOCHS,
    min_lr: float = MIN_LR,
    grad_clip: float = GRAD_CLIP,
    checkpoint_dir: str = CHECKPOINT_DIR,
    early_stopping_patience: int = None,
    random_seed: int = 42,
    save_prefix: str = "crl_encoder"
) -> tuple:
    """
    Pretrain encoder using contrastive learning with NT-Xent loss.

    Implements pretraining strategy from Mohsenvand et al. (2020) adapted for HBN dataset:
    - Learning rate warmup (linear) + cosine annealing to min_lr
    - Gradient clipping to prevent exploding gradients in LSTM
    - Validation monitoring and checkpointing (best + last)
    - Optional early stopping based on validation loss plateau

    Args:
        model: EEGContrastiveModel instance (Encoder + Projector)
        train_dataset: Training dataset (ContrastiveShardDataset or ContrastiveEEGDataset)
        val_dataset: Validation dataset (optional but recommended)
        epochs: Number of training epochs (default: 200)
        batch_size: Batch size for training (default: 256)
        lr: Peak learning rate (default: 3e-4)
        weight_decay: L2 regularization (default: 1e-6)
        device: Device to train on ('cuda' or 'cpu')
        warmup_epochs: Linear LR warmup epochs (default: 10)
        min_lr: Minimum LR for cosine annealing (default: 1e-6)
        grad_clip: Max gradient norm for clipping (default: 1.0)
        checkpoint_dir: Directory to save checkpoints
        early_stopping_patience: Stop if val loss doesn't improve for N epochs (None = no early stopping)
        random_seed: Random seed for reproducibility
        save_prefix: Prefix for checkpoint filenames (default: "crl_encoder")

    Returns:
        model: Pretrained model
        history: Dictionary with training metrics:
            - 'train_loss': List of average training losses per epoch
            - 'val_loss': List of validation losses per epoch
            - 'grad_norms': List of gradient norms per epoch
            - 'learning_rates': List of learning rates per epoch

    Example:
        >>> from contrastive_learning import EEGContrastiveModel, pretrain_contrastive
        >>> from datasets_loader_classes.shard_crl_dataset import ContrastiveShardDataset
        >>>
        >>> train_data = ContrastiveShardDataset("datasets/crl_pretraining_data_shard_*.pkl")
        >>> val_data = ContrastiveShardDataset("datasets/crl_pretraining_data_shard_*.pkl", is_train=False)
        >>> model = EEGContrastiveModel()
        >>> model, history = pretrain_contrastive(model, train_data, val_data, epochs=200)
    """
    # Set random seeds for reproducibility
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)

    # Setup device
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    print(f"Training on device: {device}")

    # Create checkpoint directory
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Setup data loaders
    # Note: For IterableDataset (shards), shuffle is not needed
    # num_workers=0 to avoid memory issues with augmentations in worker processes
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=NUM_WORKERS, 
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = None
    if val_dataset:
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            num_workers=NUM_WORKERS,
            pin_memory=True if device.type == 'cuda' else False
        )

    # Setup loss and optimizer
    criterion = NTXentLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Learning rate scheduler: warmup + cosine annealing
    def lr_lambda(current_epoch):
        if current_epoch < warmup_epochs:
            # Linear warmup: 0 → 1
            return float(current_epoch + 1) / float(warmup_epochs)
        else:
            # Cosine annealing: 1 → min_lr/lr
            if epochs <= warmup_epochs:
                return 1.0
            progress = float(current_epoch - warmup_epochs) / float(epochs - warmup_epochs)
            cosine_decay = 0.5 * (1.0 + np.cos(np.pi * progress))
            # Scale to [min_lr/lr, 1.0]
            return (min_lr / lr) + cosine_decay * (1.0 - min_lr / lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Initialize tracking
    history = {
        'train_loss': [],
        'val_loss': [],
        'grad_norms': [],
        'learning_rates': []
    }

    best_val_loss = float('inf')
    epochs_without_improvement = 0

    # Print training summary
    print(f"\n{'='*60}")
    print("Starting Contrastive Pretraining")
    print(f"{'='*60}")
    print(f"  Epochs:             {epochs}")
    print(f"  Batch size:         {batch_size}")
    print(f"  Learning rate:      {lr:.2e} (warmup + cosine → {min_lr:.2e})")
    print(f"  Weight decay:       {weight_decay:.2e}")
    print(f"  Temperature:        {criterion.temperature}")
    print(f"  Gradient clipping:  {grad_clip}")
    print(f"  Warmup epochs:      {warmup_epochs}")
    if val_dataset:
        print(f"  Early stopping:     {early_stopping_patience if early_stopping_patience else 'Disabled'}")
    print(f"  Checkpoint dir:     {checkpoint_dir}")
    print(f"  Save prefix:        {save_prefix}")
    print(f"{'='*60}\n")

    # Training loop
    for epoch in range(epochs):
        # ========== TRAINING PHASE ==========
        model.train()
        epoch_loss = 0.0
        epoch_grad_norm = 0.0
        num_batches = 0

        train_pbar = tqdm(train_loader,
            desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)

        for batch_idx, (view1, view2) in enumerate(train_pbar):
            view1 = view1.to(device, non_blocking=True)
            view2 = view2.to(device, non_blocking=True)

            # Forward pass
            _, proj1 = model(view1)  # Discard encoder output, use projections
            _, proj2 = model(view2)

            # Compute contrastive loss
            loss = criterion(proj1, proj2)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (prevents exploding gradients in LSTM)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

            optimizer.step()

            # Track metrics
            epoch_loss += loss.item()
            epoch_grad_norm += grad_norm.item()
            num_batches += 1

            # Update progress bar
            train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

        # Average metrics over batches
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_grad_norm = epoch_grad_norm / num_batches if num_batches > 0 else 0.0
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(avg_train_loss)
        history['grad_norms'].append(avg_grad_norm)
        history['learning_rates'].append(current_lr)

        # Step scheduler
        scheduler.step()

        # ========== VALIDATION PHASE ==========
        avg_val_loss = None
        if val_loader is not None:
            model.eval()
            val_loss = 0.0
            num_val_batches = 0

            val_pbar = tqdm(
                val_loader,
                desc=f"Epoch {epoch+1}/{epochs} [Val]",
                leave=False
            )

            with torch.no_grad():
                for view1, view2 in val_pbar:
                    view1 = view1.to(device, non_blocking=True)
                    view2 = view2.to(device, non_blocking=True)

                    _, proj1 = model(view1)
                    _, proj2 = model(view2)

                    loss = criterion(proj1, proj2)
                    val_loss += loss.item()
                    num_val_batches += 1

                    val_pbar.set_postfix({'loss': f'{loss.item():.4f}'})

            avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0.0
            history['val_loss'].append(avg_val_loss)

            # ========== CHECKPOINTING ==========
            # Save best model based on validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_without_improvement = 0

                checkpoint_path = checkpoint_dir / f'{save_prefix}_best.pth'
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': avg_train_loss,
                    'val_loss': avg_val_loss,
                    'best_val_loss': best_val_loss,
                    'config': CRL_CONFIG
                }, checkpoint_path)
            else:
                epochs_without_improvement += 1

            # ========== EARLY STOPPING ==========
            if early_stopping_patience and epochs_without_improvement >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
                print(f"   Validation loss has not improved for {early_stopping_patience} epochs")
                break

        # ========== LOGGING ==========
        log_msg = f"Epoch [{epoch+1:3d}/{epochs}] - Loss: {avg_train_loss:.4f}"
        if val_loader is not None and avg_val_loss is not None:
            log_msg += f" | Val: {avg_val_loss:.4f}"
            if avg_val_loss == best_val_loss:
                log_msg += " ✓"
        log_msg += f" | LR: {current_lr:.6f} | Grad: {avg_grad_norm:.4f}"
        print(log_msg)

    # ========== SAVE FINAL MODEL ==========
    final_checkpoint_path = checkpoint_dir / f'{save_prefix}_last.pth'
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': avg_train_loss,
        'val_loss': avg_val_loss if val_loader is not None else None,
        'best_val_loss': best_val_loss if val_loader is not None else None,
        'config': CRL_CONFIG
    }, final_checkpoint_path)

    # Print summary
    print(f"\n{'='*60}")
    print("✓ Pretraining Completed!")
    print(f"{'='*60}")
    print(f"  Final train loss: {avg_train_loss:.4f}")
    if val_loader is not None and avg_val_loss is not None:
        print(f"  Final val loss:   {avg_val_loss:.4f}")
        print(f"  Best val loss:    {best_val_loss:.4f}")
    print(f"\n  Saved checkpoints:")
    print(f"    - {save_prefix}_best.pth (best validation loss)")
    print(f"    - {save_prefix}_last.pth (final epoch)")
    print(f"{'='*60}\n")

    return model, history


def load_pretrained_crl_model(
    checkpoint_path: str,
    model_class,
    device: str = 'cuda',
    load_full_model: bool = True
):
    """
    Load pretrained CRL model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file (.pth)
        model_class: Model class to instantiate (EEGContrastiveModel)
        device: Device to load model on
        load_full_model: If True, load full model (encoder+projector).
                        If False, load only encoder.

    Returns:
        model: Loaded model (or encoder only)
        checkpoint: Full checkpoint dictionary with metadata

    Example:
        >>> from contrastive_learning import EEGContrastiveModel, load_pretrained_crl_model
        >>> model, ckpt = load_pretrained_crl_model(
        ...     "saved_models/crl_encoder_best.pth",
        ...     EEGContrastiveModel
        ... )
        >>> encoder = model.get_encoder()  # Extract encoder for downstream tasks
    """
    device = torch.device(device if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Instantiate model (use config from checkpoint if available)
    if 'config' in checkpoint:
        config = checkpoint['config']
        model = model_class(
            in_channels=config['n_chans'],
            n_samples=config['samplepoints'],
            output_dim=config['projector_output_dim']
        )
    else:
        # Fallback to default config
        model = model_class()

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    print(f"✓ Loaded pretrained model from {checkpoint_path}")
    print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"  Train loss: {checkpoint.get('train_loss', 'unknown')}")
    if 'val_loss' in checkpoint and checkpoint['val_loss'] is not None:
        print(f"  Val loss: {checkpoint['val_loss']}")

    if load_full_model:
        return model, checkpoint
    else:
        # Return encoder only
        return model.get_encoder(), checkpoint
