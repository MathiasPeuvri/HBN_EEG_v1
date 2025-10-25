import pickle
import numpy as np
from pathlib import Path
import pandas as pd
# Charger un shard
pkl_file = Path("datasets/chall1_clickcentered/R1_clickcentered.pkl")
with open(pkl_file, "rb") as f:
    data = pickle.load(f)
    data = pd.DataFrame(data)
print(f'data shape: {data.shape}')
print(f'data columns: {data.columns}')
print(f'data head: {data.head()}')
# Convertir avec votre fonction
from src.test_pkl_maxime import convert_maximv2_format_with_window_augmentation
converted = convert_maximv2_format_with_window_augmentation(data)

# Analyser rt_idx
print(f"rt_idx stats:")
print(f"  Min: {converted['rt_idx'].min()}")
print(f"  Max: {converted['rt_idx'].max()}")
print(f"  Mean: {converted['rt_idx'].mean():.1f}")
print(f"  Std: {converted['rt_idx'].std():.1f}")
print(f"  Distribution:")
print(converted['rt_idx'].describe())

import matplotlib.pyplot as plt
targets = converted['rt_idx']
plt.figure(figsize=(10, 5))

plt.subplot(1, 1, 1)
plt.hist(targets, bins=30, edgecolor='black', alpha=0.7)
plt.axvline(np.mean(targets), color='red', linestyle='--', label=f'Mean: {np.mean(targets):.3f}')
plt.axvline(np.median(targets), color='green', linestyle='--', label=f'Median: {np.median(targets):.3f}')
plt.xlabel('RT Index')
plt.ylabel('Frequency')
plt.title(f'RT Index Distribution (n={len(targets)})')
plt.legend()
plt.grid(alpha=0.3)


plt.tight_layout()
plt.savefig('rt_index_distribution.png', dpi=150)
print(f"\nSaved: rt_index_distribution.png")
plt.show()