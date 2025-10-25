"""Downstream regression for response time prediction"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error
from . import config
from .data_loader_ml import create_dataloaders
from ..preprocessing.preprocessor import EEGPreprocessor


class SimpleCNN1DRegressor(nn.Module):
    """Simple 3-layer 1D CNN for regression from scratch"""
    def __init__(self, dropout=config.CLS_DROPOUT):
        super().__init__()

        # Preprocessor: average reference + zscore normalization
        self.preprocessor = EEGPreprocessor(zscore_method='channel_wise')

        # Conv layer 1: 129 -> 64 channels
        self.conv1 = nn.Sequential(
            nn.Conv1d(config.NUM_CHANNELS, config.CONV1_OUT_CHANNELS,
                     kernel_size=config.KERNEL_SIZE, stride=1, padding=2),
            nn.BatchNorm1d(config.CONV1_OUT_CHANNELS),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Conv layer 2: 64 -> 32 channels
        self.conv2 = nn.Sequential(
            nn.Conv1d(config.CONV1_OUT_CHANNELS, config.CONV2_OUT_CHANNELS,
                     kernel_size=config.KERNEL_SIZE, stride=1, padding=2),
            nn.BatchNorm1d(config.CONV2_OUT_CHANNELS),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Conv layer 3: 32 -> 16 channels
        self.conv3 = nn.Sequential(
            nn.Conv1d(config.CONV2_OUT_CHANNELS, config.CONV3_OUT_CHANNELS,
                     kernel_size=config.KERNEL_SIZE, stride=1, padding=2),
            nn.BatchNorm1d(config.CONV3_OUT_CHANNELS),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Regression head
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(config.CONV3_OUT_CHANNELS, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        x = self.preprocessor(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.global_pool(x)
        x = self.regressor(x)
        return x.squeeze(-1)  # Remove last dimension



def regression_train_epoch(model, train_loader, optimizer, criterion, epoch, target_mean, target_std, epochs_n=config.CLS_EPOCHS):
    """Train for one epoch with regression metrics"""
    model.train()
    total_loss = 0
    all_preds = []
    all_targets = []

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs_n} [Train]", leave=False)

    for batch_idx, (data, targets) in enumerate(progress_bar):
        data, targets = data.to(config.DEVICE), targets.to(config.DEVICE)
        targets = (targets - target_mean) / target_std # normalize target to mean 0 and std 1
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
    #train_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    train_rmse = root_mean_squared_error(all_targets, all_preds) / np.std(all_targets)
    train_r2 = r2_score(all_targets, all_preds)
    
    # For IterableDataset, we need to count batches manually
    return total_loss / max(batch_idx + 1, 1), train_mae, train_rmse, train_r2


def validate_regression(model, val_loader, criterion, target_mean, target_std, verbose=config.VERBOSE):
    """Validate the regression model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    batch_count = 0
    with torch.no_grad():
        for data, targets in tqdm(val_loader, desc='Validation', leave=False):
            data, targets = data.to(config.DEVICE), targets.to(config.DEVICE)
            targets = (targets - target_mean) / target_std # normalize target to mean 0 and std 1
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            # Track metrics
            total_loss += loss.item()
            all_preds.extend(outputs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            batch_count += 1
    val_mae = mean_absolute_error(all_targets, all_preds)
    #val_rmse = np.sqrt(mean_squared_error(all_targets, all_preds))
    val_rmse = root_mean_squared_error(all_targets, all_preds) / np.std(all_targets)
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


def train_simple_cnn(target_column='externalizing',
                     dataset_type='regression',
                     epochs=config.CLS_EPOCHS,
                     batch_size=config.CLS_BATCH_SIZE):
    """
    Train simple CNN1D regressor from scratch.

    Args:
        target_column: Target column to predict
        dataset_type: Dataset type ('regression')
        epochs: Number of training epochs
        batch_size: Batch size

    Returns:
        model: Trained regression model
        metrics: Training metrics (losses, MAEs, RMSEs, R²s)
    """
    print("=" * 50)
    print("Simple CNN1D Regression Training")
    print(f"  Target: {target_column}")
    print("=" * 50)

    # Set config for regression task
    config.TASK_TYPE = 'regression'
    config.TARGET_COLUMN = target_column
    config.TARGET_EVENTS = None

    # Create dataloaders
    if target_column == 'response_time':
        config.DOWNSTREAM_DATA_PATTERN = config.DOWNSTREAM_CHALL1_PATTERN
        train_loader, val_loader = create_dataloaders(
            dataset_type=dataset_type, batch_size=batch_size, data_format="v1")
    else:
        train_loader, val_loader = create_dataloaders(
            dataset_type=dataset_type, batch_size=batch_size)

    # Compute target normalization stats
    print("Computing target normalization stats...")
    all_targets = []
    for _, targets in train_loader:
        all_targets.extend(targets.numpy())
    target_mean = torch.tensor(np.mean(all_targets), device=config.DEVICE)
    target_std = torch.tensor(np.std(all_targets), device=config.DEVICE)
    print(f"Target normalization: mean={target_mean:.3f}, std={target_std:.3f}")

    # Create model
    model = SimpleCNN1DRegressor().to(config.DEVICE)
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {trainable_params:,}")

    # Initialize optimizer and criterion
    optimizer = optim.Adam(model.parameters(), lr=config.CLS_LEARNING_RATE,
                          weight_decay=config.CLS_WEIGHT_DECAY)
    criterion = nn.MSELoss()
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
            model, train_loader, optimizer, criterion, epoch, target_mean, target_std, epochs_n=epochs)
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        train_rmses.append(train_rmse)
        train_r2s.append(train_r2)

        # Validate
        val_loss, val_mae, val_rmse, val_r2 = validate_regression(model, val_loader, criterion, target_mean, target_std)
        val_losses.append(val_loss)
        val_maes.append(val_mae)
        val_rmses.append(val_rmse)
        val_r2s.append(val_r2)

        # Update learning rate
        scheduler.step(val_mae)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Train MAE: {train_mae:.6f}, Train RMSE: {train_rmse:.6f}, "
              f"Val Loss: {val_loss:.6f}, Val MAE: {val_mae:.6f}, Val RMSE: {val_rmse:.6f}, Val R²: {val_r2:.6f}")

        # Save best model
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            model_name = f"simple_cnn_regressor_{target_column}_best.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_mae': train_mae,
                'val_mae': val_mae,
                'val_r2': val_r2,
            }, config.MODEL_DIR / model_name)
            print(f"Saved best model with val_mae: {val_mae:.3f}")

    print("\n" + "=" * 50)
    print(f"Training Complete! Best validation MAE: {best_val_mae:.3f}")
    model_name = f"simple_cnn_regressor_{target_column}_best.pth"
    print(f"Model saved to: {config.MODEL_DIR / model_name}")

    # Load and validate best model
    print("\nFinal validation with best model:")
    model.load_state_dict(torch.load(config.MODEL_DIR / model_name, weights_only=False)['model_state_dict'])
    validate_regression(model, val_loader, criterion, target_mean, target_std, verbose=True)
    print("=" * 50)

    return model, (train_losses, val_losses, train_maes, val_maes, train_rmses, val_rmses, train_r2s, val_r2s)


def main():
    parser = argparse.ArgumentParser(
        description='Train simple CNN1D regressor from scratch',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train simple CNN for response time prediction
  python simple_regression.py --target response_time

  # Train with custom epochs and batch size
  python simple_regression.py --target response_time --epochs 100 --batch-size 32
        """
    )
    parser.add_argument('--target', type=str, default='response_time',
                       help='Target column to predict (default: response_time)')
    parser.add_argument('--epochs', type=int, default=config.CLS_EPOCHS,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=config.CLS_BATCH_SIZE,
                       help='Batch size for training')
    args = parser.parse_args()

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # Train model
    model, metrics = train_simple_cnn(
        target_column=args.target,
        epochs=args.epochs,
        batch_size=args.batch_size)

    # Print final statistics
    train_losses, val_losses, train_maes, val_maes, train_rmses, val_rmses, train_r2s, val_r2s = metrics
    print(f"\nFinal Training MAE: {train_maes[-1]:.3f}")
    print(f"Final Validation MAE: {val_maes[-1]:.3f}")
    print(f"Best Validation MAE: {min(val_maes):.3f}")
    print(f"Final Validation R²: {val_r2s[-1]:.3f}")

if __name__ == "__main__":
    main()