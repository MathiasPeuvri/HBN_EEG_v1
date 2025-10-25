import numpy as np
import pandas as pd
import pickle
import glob
import matplotlib.pyplot as plt
from src.ML_pipeline_test import config
from src.ML_pipeline_test.datasets_loader_classes.shard_downstream_dataset import SequentialShardDownstreamDataset

# Configure
config.TASK_TYPE = 'regression'
config.TARGET_COLUMN = 'externalizing'
config.TARGET_EVENTS = None
shard_pattern = config.DOWNSTREAM_DATA_PATTERN

print(f"Data pattern: {shard_pattern}")
print(f"Target: {config.TARGET_COLUMN}")

# Load dataset to get psychopathology mapping
dataset = SequentialShardDownstreamDataset(
    shard_pattern=shard_pattern,
    task_type='regression',
    is_train=True
)

# Get train shards
shard_files = sorted(glob.glob(shard_pattern))
train_shards = [f for f in shard_files if 'R2' not in f]

print(f"\nTrain shards: {len(train_shards)}")

# Collect externalizing values from first shard only (for speed)
targets = []
missing = 0
valid = 0

for train_shard in train_shards:
    with open(train_shard, 'rb') as f:
        data = pickle.load(f)
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        for idx in range(len(data)):
            subject_id = data.iloc[idx]['subject']
            if isinstance(subject_id, pd.Series):
                subject_id = subject_id.iloc[0]

            if subject_id in dataset.psycho_factor_map:
                target = dataset.psycho_factor_map[subject_id]
                if not pd.isna(target):
                    targets.append(target)
                    valid += 1
            else:
                missing += 1

targets = np.array(targets)

# Print statistics
print(f"\nStatistics (from 1 shard):")
print(f"  Valid samples: {valid}")
print(f"  Missing subjects: {missing}")
print(f"  Mean: {np.mean(targets):.4f}")
print(f"  Std: {np.std(targets):.4f}")
print(f"  Min: {np.min(targets):.4f}")
print(f"  Max: {np.max(targets):.4f}")
print(f"  Median: {np.median(targets):.4f}")

# Plot distribution
plt.figure(figsize=(10, 5))

plt.subplot(1, 1, 1)
plt.hist(targets, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(targets), color='red', linestyle='--', label=f'Mean: {np.mean(targets):.3f}')
plt.axvline(np.median(targets), color='green', linestyle='--', label=f'Median: {np.median(targets):.3f}')
plt.xlabel('Externalizing Score')
plt.ylabel('Frequency')
plt.title(f'Externalizing Distribution (n={len(targets)})')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('externalizing_distribution.png', dpi=150)
print(f"\nSaved: externalizing_distribution.png")
plt.show()