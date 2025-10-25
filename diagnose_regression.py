"""Quick diagnostic script to understand regression failure"""
import torch
import numpy as np
from src.ML_pipeline_test import config
from src.ML_pipeline_test.data_loader_ml import create_dataloaders

# Setup
config.TASK_TYPE = 'regression'
config.TARGET_COLUMN = 'externalizing'
config.TARGET_EVENTS = None

print("Loading data...")
train_loader, val_loader = create_dataloaders(dataset_type='regression', batch_size=512)

# Collect first 5000 samples
print("Analyzing first 5000 train samples...")
all_signals = []
all_targets = []
for i, (signals, targets) in enumerate(train_loader):
    all_signals.append(signals.numpy())
    all_targets.append(targets.numpy())
    if len(all_targets) * 512 >= 5000:
        break

signals = np.concatenate(all_signals)[:5000]
targets = np.concatenate(all_targets)[:5000]

print(f"\n=== DATA STATISTICS ===")
print(f"Signals shape: {signals.shape}")
print(f"Signal mean: {signals.mean():.6f}, std: {signals.std():.6f}")
print(f"Signal min: {signals.min():.6f}, max: {signals.max():.6f}")
print(f"\nTarget stats:")
print(f"  Mean: {targets.mean():.3f}, Std: {targets.std():.3f}")
print(f"  Min: {targets.min():.3f}, Max: {targets.max():.3f}")
print(f"  Unique values: {len(np.unique(targets))}")

# Check if all signals look the same (no variance)
signal_variance_per_channel = signals.var(axis=(0, 2))  # variance across samples and time
print(f"\nSignal variance per channel (first 10):")
print(signal_variance_per_channel[:10])
print(f"Channels with near-zero variance: {(signal_variance_per_channel < 1e-6).sum()}/{len(signal_variance_per_channel)}")

# Check correlation
from scipy.stats import pearsonr
signal_features = signals.mean(axis=(1, 2))  # Simple feature: mean across channels and time
corr, pval = pearsonr(signal_features, targets)
print(f"\nCorrelation between signal mean and target: {corr:.4f} (p={pval:.4e})")

print("\n=== DIAGNOSIS ===")
if abs(corr) < 0.05:
    print("⚠️  PROBLEM: Signal features have almost ZERO correlation with target!")
    print("   → The EEG signals may not contain useful information for this task")
    print("   → Or features need better engineering (current: just mean)")
else:
    print(f"✓ Correlation exists ({corr:.3f}), model should be able to learn")

if (signal_variance_per_channel < 1e-6).sum() > 50:
    print("⚠️  PROBLEM: Many channels have zero variance")
    print("   → Data preprocessing issue or dead channels")

print("\n=== RECOMMENDATIONS ===")
if abs(corr) < 0.05:
    print("1. Try multi-task learning (predict all psychopathology factors)")
    print("2. Use pretrained encoder from challenge 1")
    print("3. Check if other factors (p_factor, internalizing) have better correlation")
else:
    print("1. Increase model capacity (more layers/channels)")
    print("2. Use better loss (Huber loss for robustness)")
    print("3. Increase learning rate or use different optimizer")