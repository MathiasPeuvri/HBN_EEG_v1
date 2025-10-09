"""Downstream regression for response time prediction"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from . import config
from .data_loader_ml import create_dataloaders
from .models import CNN1DAutoencoder, BinaryClassifier

"""
Regression pipeline for predicting continuous values (e.g., response_time)
Based on classification.py but adapted for regression tasks:
1. Uses MSE loss instead of CrossEntropy
2. Uses MAE, RMSE, R² metrics instead of accuracy
3. Outputs single continuous value instead of class probabilities
4. Model architecture adapted for regression output

Supports two encoder types:
- 'autoencoder': Original masked autoencoder pretraining
- 'crl': Contrastive Representation Learning pretraining
"""


def load_pretrained_encoder(encoder_type='autoencoder', autoencoder_class=CNN1DAutoencoder):
    """
    Load pretrained encoder (either autoencoder or CRL).

    Args:
        encoder_type: Type of encoder ('autoencoder' or 'crl')
        autoencoder_class: Autoencoder class (only used if encoder_type='autoencoder')

    Returns:
        encoder: Pretrained encoder module or None if loading fails
    """
    if encoder_type == 'autoencoder':
        # Load masked autoencoder encoder
        try:
            checkpoint = torch.load(config.AUTOENCODER_PATH, map_location=config.DEVICE, weights_only=False)
            autoencoder = autoencoder_class()
            autoencoder.load_state_dict(checkpoint['model_state_dict'])
            print(f"✓ Loaded pretrained autoencoder from epoch {checkpoint['epoch']}")
            return autoencoder.encoder
        except FileNotFoundError:
            print("⚠ Warning: No pretrained autoencoder found. Using random initialization.")
            return None

    elif encoder_type == 'crl':
        # Load CRL encoder
        try:
            from .contrastive_learning import EEGContrastiveModel

            # Try to find CRL checkpoint
            crl_checkpoint_path = config.MODEL_DIR / "crl_encoder_best.pth"
            if not crl_checkpoint_path.exists():
                # Fallback to last checkpoint
                crl_checkpoint_path = config.MODEL_DIR / "crl_encoder_last.pth"

            if not crl_checkpoint_path.exists():
                raise FileNotFoundError(f"No CRL checkpoint found at {config.MODEL_DIR}")

            checkpoint = torch.load(crl_checkpoint_path, map_location=config.DEVICE, weights_only=False)

            # Create full CRL model
            crl_config = checkpoint.get('config', {})
            model = EEGContrastiveModel(
                in_channels=crl_config.get('n_chans', 129),
                n_samples=crl_config.get('samplepoints', 200),
                output_dim=crl_config.get('projector_output_dim', 128)
            )
            model.load_state_dict(checkpoint['model_state_dict'])

            print(f"✓ Loaded CRL encoder from epoch {checkpoint['epoch']}")
            print(f"  Checkpoint: {crl_checkpoint_path.name}")
            if 'val_loss' in checkpoint:
                print(f"  Val loss: {checkpoint['val_loss']:.4f}")

            # Return encoder only (discard projector)
            return model.get_encoder()

        except FileNotFoundError as e:
            print(f"⚠ Warning: {e}")
            print("  Run CRL pretraining first: python crl_pretraining.py")
            print("  Using random initialization.")
            return None
        except ImportError as e:
            print(f"⚠ Warning: Could not import CRL module: {e}")
            print("  Using random initialization.")
            return None

    else:
        raise ValueError(f"Unknown encoder_type: {encoder_type}. Choose 'autoencoder' or 'crl'.")


class RegressionHead(nn.Module):
    """Regression head for autoencoder-based continuous value prediction"""
    def __init__(self, encoder=None, freeze_encoder=True, dropout=config.CLS_DROPOUT):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder

        if encoder is not None and freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False

        # Feature extraction (same as BinaryClassifier)
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten()
        )

        # Regression head (similar to BinaryClassifier but with 1 output)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(config.CONV3_OUT_CHANNELS, 32),  # 16 -> 32
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # Single output for regression
        )

    def forward(self, x):
        if self.encoder is not None:
            features = self.encoder(x) # Extract features
        else:
            features = x
        # Apply feature extraction (pooling + flattening) then Regress to single value
        features = self.feature_extractor(features)
        return self.regressor(features).squeeze(-1)  # Remove last dimension


class CRLRegressionHead(nn.Module):
    """Regression head for CRL encoder (uses bi-LSTM Projector in regression mode)"""
    def __init__(self, encoder, freeze_encoder=True, dropout=config.CLS_DROPOUT):
        super().__init__()
        self.encoder = encoder

        if freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False

        # Import Projector and use it in regression mode
        from .contrastive_learning.models import Projector
        self.projector = Projector(
            input_dim=4,  # CRL encoder outputs 4 channels
            output_dim=1,  # Scalar output for regression
            task_mode='regression'
        )

    def forward(self, x):
        features = self.encoder(x)  # (batch, 4, time_reduced)
        output = self.projector(features)  # (batch,) - squeeze done in Projector
        return output


def regression_train_epoch(model, train_loader, optimizer, criterion, epoch, epochs_n=config.CLS_EPOCHS):
    """Train for one epoch with regression metrics"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs_n} [Train]')
    
    for batch_idx, (data, targets) in enumerate(progress_bar):
        data, targets = data.to(config.DEVICE), targets.to(config.DEVICE)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(targets.cpu().numpy())
        
        # Update progress bar
        if batch_idx % config.LOG_INTERVAL == 0:
            avg_loss = total_loss / (batch_idx + 1)
            mae = mean_absolute_error(all_targets, all_preds)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'mae': f'{mae:.3f}'})
    
    train_mae = mean_absolute_error(all_targets, all_preds)
    train_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    train_r2 = r2_score(all_targets, all_preds)
    
    # For IterableDataset, we need to count batches manually
    return total_loss / max(batch_idx + 1, 1), train_mae, train_rmse, train_r2


def validate_regression(model, val_loader, criterion, verbose=config.VERBOSE):
    """Validate the regression model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    batch_count = 0
    
    with torch.no_grad():
        for data, targets in tqdm(val_loader, desc='Validation'):
            data, targets = data.to(config.DEVICE), targets.to(config.DEVICE)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Track metrics
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            batch_count += 1
    
    val_mae = mean_absolute_error(all_targets, all_preds)
    val_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    val_r2 = r2_score(all_targets, all_preds)
    
    # Print detailed metrics
    if verbose:
        print(f"\nValidation Metrics:")
        print(f"MAE: {val_mae:.3f}")
        print(f"RMSE: {val_rmse:.3f}")
        print(f"R² Score: {val_r2:.3f}")
        print(f"Target range: [{np.min(all_targets):.3f}, {np.max(all_targets):.3f}]")
        print(f"Prediction range: [{np.min(all_preds):.3f}, {np.max(all_preds):.3f}]")
    
    # For IterableDataset, we need to count batches manually
    return total_loss / max(batch_count, 1), val_mae, val_rmse, val_r2


def train_regressor(encoder_type='autoencoder',
                   autoencoder_class=CNN1DAutoencoder,
                   target_column=None,
                   dataset_type='regression',
                   epochs=config.CLS_EPOCHS,
                   batch_size=config.CLS_BATCH_SIZE,
                   freeze_encoder=True):
    """
    Main training function for regressor.

    Args:
        encoder_type: Type of pretrained encoder ('autoencoder' or 'crl')
        autoencoder_class: Autoencoder class (only used if encoder_type='autoencoder')
        target_column: Target column to predict
        dataset_type: Dataset type ('regression')
        epochs: Number of training epochs
        batch_size: Batch size
        freeze_encoder: Whether to freeze encoder weights

    Returns:
        model: Trained regression model
        metrics: Training metrics (losses, MAEs, RMSEs, R²s)
    """
    print("=" * 50)
    print("Starting Regression Training")
    print(f"  Encoder type: {encoder_type}")
    print("=" * 50)

    # Set config for regression task
    config.TASK_TYPE = 'regression'  # Ensure regression mode
    if target_column is not None:
        config.TARGET_COLUMN = target_column
    config.TARGET_EVENTS = None  # Use all data for regression

    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        dataset_type=dataset_type, batch_size=batch_size)

    # Load pretrained encoder
    encoder = load_pretrained_encoder(
        encoder_type=encoder_type,
        autoencoder_class=autoencoder_class
    )

    # Initialize model based on encoder type
    if encoder_type == 'crl':
        model = CRLRegressionHead(encoder=encoder, freeze_encoder=freeze_encoder).to(config.DEVICE)
    else:
        model = RegressionHead(encoder=encoder, freeze_encoder=freeze_encoder).to(config.DEVICE)
    
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Encoder frozen: {freeze_encoder} --> Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.CLS_LEARNING_RATE, weight_decay=config.CLS_WEIGHT_DECAY)
    criterion = nn.MSELoss()
    
    # Learning rate scheduler (monitor MAE instead of accuracy)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_mae = float('inf')
    train_losses, val_losses = [], []
    train_maes, val_maes = [], []
    train_rmses, val_rmses = [], []
    train_r2s, val_r2s = [], []
    
    for epoch in range(epochs):
        # Train
        train_loss, train_mae, train_rmse, train_r2 = regression_train_epoch(
            model, train_loader, optimizer, criterion, epoch, epochs_n=epochs)
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        train_rmses.append(train_rmse)
        train_r2s.append(train_r2)
        
        # Validate
        val_loss, val_mae, val_rmse, val_r2 = validate_regression(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        val_rmses.append(val_rmse)
        val_r2s.append(val_r2)
        
        # Update learning rate
        scheduler.step(val_mae)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f}, "
              f"Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}, Val R²: {val_r2:.6f}")
        
        # Save best model (based on MAE)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            # Include encoder type in filename
            encoder_suffix = f"_{encoder_type}" if encoder_type != 'autoencoder' else ""
            model_name = f"regressor_{target_column}{encoder_suffix}_best.pth" if target_column else f"regressor{encoder_suffix}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'val_r2': val_r2,
                'encoder_type': encoder_type,
            }, config.MODEL_DIR / model_name)
            print(f"Saved best model with val_mae: {val_mae:.3f}")
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best validation MAE: {best_val_mae:.3f}")
    encoder_suffix = f"_{encoder_type}" if encoder_type != 'autoencoder' else ""
    model_name = f"regressor_{target_column}{encoder_suffix}_best.pth" if target_column else f"regressor{encoder_suffix}_best.pth"
    print(f"Model saved to: {config.MODEL_DIR / model_name}")
    
    # Load and validate best model
    print("\nFinal validation with best model:")
    model.load_state_dict(torch.load(config.MODEL_DIR / model_name, weights_only=False)['model_state_dict'])
    validate_regression(model, val_loader, criterion, verbose=True)
    print("=" * 50)
    
    return model, (train_losses, val_losses, train_maes, val_maes, train_rmses, val_rmses, train_r2s, val_r2s)


def main():
    parser = argparse.ArgumentParser(
        description='Train regressor with pretrained encoder',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train with autoencoder (default)
  python regression.py --target response_time

  # Train with CRL encoder
  python regression.py --encoder_type crl --target response_time

  # Fine-tune encoder
  python regression.py --encoder_type crl --target response_time --unfreeze
        """
    )
    parser.add_argument('--encoder_type', type=str, default='autoencoder',
                       choices=['autoencoder', 'crl'],
                       help='Type of pretrained encoder (default: autoencoder)')
    parser.add_argument('--epochs', type=int, default=config.CLS_EPOCHS,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=config.CLS_BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--unfreeze', action='store_true',
                       help='Unfreeze encoder for fine-tuning')
    parser.add_argument('--target', type=str, default=config.TARGET_COLUMN,
                       help='Target column to predict')
    args = parser.parse_args()

    torch.manual_seed(config.RANDOM_SEED); np.random.seed(config.RANDOM_SEED)

    # Train model
    model, metrics = train_regressor(
        encoder_type=args.encoder_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        freeze_encoder=not args.unfreeze,
        target_column=args.target)
    
    # Print final statistics
    train_losses, val_losses, train_maes, val_maes, train_rmses, val_rmses, train_r2s, val_r2s = metrics
    print(f"\nFinal Training MAE: {train_maes[-1]:.3f}")
    print(f"Final Validation MAE: {val_maes[-1]:.3f}")
    print(f"Best Validation MAE: {min(val_maes):.3f}")
    print(f"Final Validation R²: {val_r2s[-1]:.3f}")

if __name__ == "__main__":
    main()