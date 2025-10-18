"""
CRL Pretraining Script for HBN EEG Analysis

Main CLI script for contrastive representation learning pretraining.
Loads multi-task sharded data and trains encoder using NT-Xent loss.

Usage:
    # Basic usage with default parameters
    python crl_pretraining.py

    # Custom configuration
    python crl_pretraining.py \\
        --epochs 200 \\
        --batch-size 256 \\
        --lr 3e-4 \\
        --device cuda

    # Resume from checkpoint
    python crl_pretraining.py \\
        --resume saved_models/crl_encoder_last.pth
"""

import argparse
import sys
from pathlib import Path

import torch

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.ML_pipeline_test.contrastive_learning import (
    EEGContrastiveModel, pretrain_contrastive, CRL_CONFIG)
from src.ML_pipeline_test.datasets_loader_classes.shard_crl_dataset import (
    ContrastiveShardDataset)
from src.ML_pipeline_test import config as ml_config


def main():
    parser = argparse.ArgumentParser(
        description="Pretrain EEG encoder using Contrastive Representation Learning (CRL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # Train with default parameters (200 epochs, batch 256)
        python crl_pretraining.py

        # Custom training configuration
        python crl_pretraining.py --epochs 300 --batch-size 128 --lr 1e-3

        # Enable early stopping
        python crl_pretraining.py --early-stopping 20

        # Resume from checkpoint
        python crl_pretraining.py --resume saved_models/crl_encoder_last.pth
        """
    )

    # Data arguments
    parser.add_argument("--data-pattern", type=str,
        default=str(ml_config.DATA_DIR / "crl_pretraining_data_shard_*.pkl"),
        help="Glob pattern for CRL shard files")

    parser.add_argument("--train-split", type=float, default=0.8,
        help="Train/val split ratio (default: 0.8)")

    # Model arguments
    parser.add_argument("--in-channels", type=int, default=CRL_CONFIG['n_chans'],
        help=f"Number of input channels (default: {CRL_CONFIG['n_chans']})")

    parser.add_argument("--n-samples", type=int, default=CRL_CONFIG['samplepoints'],
        help=f"Number of time samples (default: {CRL_CONFIG['samplepoints']})")

    parser.add_argument("--output-dim", type=int, default=CRL_CONFIG['projector_output_dim'],
        help=f"Projection dimension (default: {CRL_CONFIG['projector_output_dim']})")

    # Training arguments
    parser.add_argument("--epochs", type=int, default=CRL_CONFIG['epochs'],
        help=f"Number of training epochs (default: {CRL_CONFIG['epochs']})")
        
    parser.add_argument("--batch-size", type=int, default=CRL_CONFIG['batch_size'],
        help=f"Batch size (default: {CRL_CONFIG['batch_size']})")

    parser.add_argument("--lr", type=float, default=CRL_CONFIG['learning_rate'],
        help=f"Learning rate (default: {CRL_CONFIG['learning_rate']})")

    parser.add_argument("--weight-decay", type=float, default=CRL_CONFIG['weight_decay'],
        help=f"Weight decay (default: {CRL_CONFIG['weight_decay']})")

    parser.add_argument("--warmup-epochs", type=int, default=CRL_CONFIG['warmup_epochs'],
        help=f"Warmup epochs (default: {CRL_CONFIG['warmup_epochs']})")

    parser.add_argument("--min-lr", type=float, default=CRL_CONFIG['min_lr'],
        help=f"Minimum LR for cosine annealing (default: {CRL_CONFIG['min_lr']})")

    parser.add_argument("--grad-clip", type=float, default=CRL_CONFIG['grad_clip'],
        help=f"Gradient clipping max norm (default: {CRL_CONFIG['grad_clip']})")

    # Regularization arguments
    parser.add_argument("--early-stopping", type=int, default=None,
        help="Early stopping patience (default: None = disabled)")

    # System arguments
    parser.add_argument("--device", type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
        choices=['cuda', 'cpu'], help="Device to train on")

    parser.add_argument("--checkpoint-dir", type=str, default=str(ml_config.MODEL_DIR),
        help=f"Checkpoint directory (default: {ml_config.MODEL_DIR})")

    parser.add_argument("--save-prefix", type=str, default="crl_encoder",
        help="Checkpoint filename prefix (default: crl_encoder)")

    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    # Resume training
    parser.add_argument("--resume", type=str, default=None,
        help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Print configuration
    print("\n" + "="*70)
    print("  CRL Pretraining Configuration")
    print("="*70)
    print(f"  Data pattern:    {args.data_pattern}")
    print(f"  Train split:     {args.train_split}")
    print(f"  In channels:     {args.in_channels}")
    print(f"  Samples:         {args.n_samples}")
    print(f"  Projection dim:  {args.output_dim}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size}")
    print(f"  Learning rate:   {args.lr}")
    print(f"  Warmup epochs:   {args.warmup_epochs}")
    print(f"  Device:          {args.device}")
    print(f"  Checkpoint dir:  {args.checkpoint_dir}")
    if args.resume:
        print(f"  Resume from:     {args.resume}")
    print("="*70 + "\n")

    # Create datasets
    print("Loading datasets...")
    try:
        # Note: train/val split is done deterministically via hash of shard filenames
        # Same shard can appear in train OR val, never both (deterministic based on filename hash)
        train_dataset = ContrastiveShardDataset(
            shard_pattern=args.data_pattern,
            train_split=args.train_split,
            is_train=True,
            seed=args.seed
        )
        val_dataset = ContrastiveShardDataset(
            shard_pattern=args.data_pattern,
            train_split=args.train_split,
            is_train=False,
            seed=args.seed
        )
    except ValueError as e:
        print(f"\nError: {e}")
        print("\nMake sure to create CRL shards first:")
        print("  python src/database_to_dataset/database_to_crl_pretraining_shards.py --verbose\n")
        sys.exit(1)

    # Create model
    print("Initializing model...")
    model = EEGContrastiveModel(
        in_channels=args.in_channels,
        n_samples=args.n_samples,
        output_dim=args.output_dim
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
        print(f"  Loaded checkpoint from epoch {start_epoch}")

    # Train model
    print("\nStarting training...\n")
    model, history = pretrain_contrastive(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        warmup_epochs=args.warmup_epochs,
        min_lr=args.min_lr,
        grad_clip=args.grad_clip,
        checkpoint_dir=args.checkpoint_dir,
        early_stopping_patience=args.early_stopping,
        random_seed=args.seed,
        save_prefix=args.save_prefix
    )

    # Print final summary
    print("\n" + "="*70)
    print("  Training Summary")
    print("="*70)
    print(f"  Final train loss:  {history['train_loss'][-1]:.4f}")
    if history['val_loss']:
        print(f"  Final val loss:    {history['val_loss'][-1]:.4f}")
        print(f"  Best val loss:     {min(history['val_loss']):.4f}")
    print(f"  Total epochs:      {len(history['train_loss'])}")
    print("="*70 + "\n")



if __name__ == "__main__":
    main()
