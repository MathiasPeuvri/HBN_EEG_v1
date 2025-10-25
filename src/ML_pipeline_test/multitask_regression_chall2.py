"""Multi-task regression for all 4 psychopathology factors"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from . import config
from .data_loader_ml import create_dataloaders
from ..preprocessing.preprocessor import EEGPreprocessor


class MultiTaskCNN1DRegressor(nn.Module):
    """CNN with shared encoder + 4 task-specific heads"""
    def __init__(self, dropout=config.CLS_DROPOUT):
        super().__init__()

        # Preprocessor
        self.preprocessor = EEGPreprocessor(zscore_method='channel_wise')

        # Shared encoder
        self.conv1 = nn.Sequential(
            nn.Conv1d(config.NUM_CHANNELS, config.CONV1_OUT_CHANNELS,
                     kernel_size=config.KERNEL_SIZE, stride=1, padding=2),
            nn.BatchNorm1d(config.CONV1_OUT_CHANNELS),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(config.CONV1_OUT_CHANNELS, config.CONV2_OUT_CHANNELS,
                     kernel_size=config.KERNEL_SIZE, stride=1, padding=2),
            nn.BatchNorm1d(config.CONV2_OUT_CHANNELS),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.conv3 = nn.Sequential(
            nn.Conv1d(config.CONV2_OUT_CHANNELS, config.CONV3_OUT_CHANNELS,
                     kernel_size=config.KERNEL_SIZE, stride=1, padding=2),
            nn.BatchNorm1d(config.CONV3_OUT_CHANNELS),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)

        # Task-specific heads (4 factors)
        self.heads = nn.ModuleDict({
            'p_factor': self._make_head(dropout),
            'externalizing': self._make_head(dropout),
            'internalizing': self._make_head(dropout),
            'attention': self._make_head(dropout)
        })

    def _make_head(self, dropout):
        return nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(config.CONV3_OUT_CHANNELS, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x, return_features=False):
        x = self.preprocessor(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        features = self.global_pool(x)

        # Predict all 4 factors
        outputs = {task: head(features).squeeze(-1) for task, head in self.heads.items()}

        return outputs if not return_features else (outputs, features)


def train_epoch(model, train_loader, optimizer, epoch, target_stats, epochs_n):
    """Train one epoch with multi-task loss"""
    model.train()
    total_loss = 0
    all_preds = {task: [] for task in target_stats.keys()}
    all_targets = {task: [] for task in target_stats.keys()}

    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs_n} [Train]", leave=False)

    for batch_idx, (data, targets_dict) in enumerate(progress_bar):
        data = data.to(config.DEVICE)

        # Normalize targets
        targets_norm = {}
        for task, (mean, std) in target_stats.items():
            t = targets_dict[task].to(config.DEVICE)
            targets_norm[task] = (t - mean) / std

        # Forward pass
        outputs = model(data)

        # Multi-task loss (average across tasks)
        loss = sum(nn.functional.mse_loss(outputs[task], targets_norm[task])
                   for task in target_stats.keys()) / len(target_stats)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        for task in target_stats.keys():
            all_preds[task].extend(outputs[task].detach().cpu().numpy())
            all_targets[task].extend(targets_norm[task].cpu().numpy())

        if batch_idx % config.LOG_INTERVAL == 0:
            progress_bar.set_postfix({'loss': f'{total_loss / (batch_idx + 1):.4f}'})

    # Compute metrics per task
    metrics = {}
    for task in target_stats.keys():
        mae = mean_absolute_error(all_targets[task], all_preds[task])
        r2 = r2_score(all_targets[task], all_preds[task])
        metrics[task] = {'mae': mae, 'r2': r2}

    return total_loss / max(batch_idx + 1, 1), metrics


def validate(model, val_loader, target_stats):
    """Validate multi-task model"""
    model.eval()
    total_loss = 0
    all_preds = {task: [] for task in target_stats.keys()}
    all_targets = {task: [] for task in target_stats.keys()}
    batch_count = 0

    with torch.no_grad():
        for data, targets_dict in tqdm(val_loader, desc='Validation', leave=False):
            data = data.to(config.DEVICE)

            # Normalize targets
            targets_norm = {}
            for task, (mean, std) in target_stats.items():
                t = targets_dict[task].to(config.DEVICE)
                targets_norm[task] = (t - mean) / std

            # Forward
            outputs = model(data)

            # Loss
            loss = sum(nn.functional.mse_loss(outputs[task], targets_norm[task])
                      for task in target_stats.keys()) / len(target_stats)

            total_loss += loss.item()
            for task in target_stats.keys():
                all_preds[task].extend(outputs[task].cpu().numpy())
                all_targets[task].extend(targets_norm[task].cpu().numpy())
            batch_count += 1

    # Metrics per task
    metrics = {}
    for task in target_stats.keys():
        mae = mean_absolute_error(all_targets[task], all_preds[task])
        r2 = r2_score(all_targets[task], all_preds[task])
        metrics[task] = {'mae': mae, 'r2': r2}

    return total_loss / max(batch_count, 1), metrics


def compute_target_stats(train_loader, tasks):
    """Compute mean/std for all tasks"""
    print("Computing normalization stats for all tasks...")
    all_targets = {task: [] for task in tasks}

    for _, targets_dict in train_loader:
        for task in tasks:
            all_targets[task].extend(targets_dict[task].numpy())

    stats = {}
    for task in tasks:
        mean = torch.tensor(np.mean(all_targets[task]), device=config.DEVICE)
        std = torch.tensor(np.std(all_targets[task]), device=config.DEVICE)
        stats[task] = (mean, std)
        print(f"  {task}: mean={mean:.3f}, std={std:.3f}")

    return stats


def train_multitask(epochs=config.CLS_EPOCHS, batch_size=config.CLS_BATCH_SIZE):
    """Train multi-task regressor"""
    from torch.utils.data import DataLoader
    from .datasets_loader_classes.shard_multitaskregression_dataset import MultiTaskDownstreamDataset

    print("=" * 50)
    print("Multi-Task CNN1D Regression Training")
    print("  Targets: p_factor, externalizing, internalizing, attention")
    print("=" * 50)

    tasks = ['p_factor', 'externalizing', 'internalizing', 'attention']

    # Create multi-task dataloaders
    train_dataset = MultiTaskDownstreamDataset(
        config.DOWNSTREAM_DATA_PATTERN, train_split=config.TRAIN_SPLIT,
        is_train=True, seed=config.RANDOM_SEED)
    val_dataset = MultiTaskDownstreamDataset(
        config.DOWNSTREAM_DATA_PATTERN, train_split=config.TRAIN_SPLIT,
        is_train=False, seed=config.RANDOM_SEED)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=2, pin_memory=config.DEVICE.type == 'cuda')
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                           num_workers=2, pin_memory=config.DEVICE.type == 'cuda')

    # Compute normalization stats
    target_stats = compute_target_stats(train_loader, tasks)

    # Create model
    model = MultiTaskCNN1DRegressor().to(config.DEVICE)
    print(f"Total trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Optimizer & scheduler
    optimizer = optim.Adam(model.parameters(), lr=config.CLS_LEARNING_RATE,
                          weight_decay=config.CLS_WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5)

    # Training loop
    best_val_loss = float('inf')
    for epoch in range(epochs):
        train_loss, train_metrics = train_epoch(model, train_loader, optimizer,
                                                epoch, target_stats, epochs)
        val_loss, val_metrics = validate(model, val_loader, target_stats)

        scheduler.step(val_loss)

        # Print per-task metrics
        print(f"\nEpoch {epoch+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
        for task in tasks:
            print(f"  {task}: Train MAE={train_metrics[task]['mae']:.3f}, "
                  f"Train R²={train_metrics[task]['r2']:.3f}, "
                  f"Val MAE={val_metrics[task]['mae']:.3f}, "
                  f"Val R²={val_metrics[task]['r2']:.3f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'val_metrics': val_metrics,
            }, config.MODEL_DIR / "multitask_regressor_best.pth")
            print(f"  Saved best model (val_loss={val_loss:.4f})")

    print("\n" + "=" * 50)
    print(f"Training Complete! Best val loss: {best_val_loss:.4f}")
    return model, (target_stats, best_val_loss)


def main():
    parser = argparse.ArgumentParser(description='Train multi-task CNN1D regressor')
    parser.add_argument('--epochs', type=int, default=config.CLS_EPOCHS)
    parser.add_argument('--batch-size', type=int, default=config.CLS_BATCH_SIZE)
    args = parser.parse_args()

    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    model, metrics = train_multitask(epochs=args.epochs, batch_size=args.batch_size)


if __name__ == "__main__":
    main()