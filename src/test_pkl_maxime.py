import pickle
from pathlib import Path
import pandas as pd
import numpy as np


def convert_maximv1_format(df):
    """Convert eval format (winwdows/vals) to standard format (signal/response_time)."""
    if 'signal' in df.columns or 'winwdows' not in df.columns:
        return df

    rows = [{'signal': row['winwdows'][i], 'response_time': row['vals'][i, 0],
             'p_factor': row['vals'][i, 1], 'attention': row['vals'][i, 2],
             'internalizing': row['vals'][i, 3], 'externalizing': row['vals'][i, 4],
             'subject': row['subject']}
            for _, row in df.iterrows() for i in range(len(row['winwdows']))]

    return pd.DataFrame(rows)


pkl_file = Path("datasets/eval/R1.pkl")
with open(pkl_file, "rb") as f: data = pickle.load(f)
data = pd.DataFrame(data)

print(f"Original: {data.shape}, columns: {data.columns.tolist()}")
print(f"winwdows[0] shape: {data['winwdows'].iloc[0].shape}, vals[0] shape: {data['vals'].iloc[0].shape}")

converted = convert_eval_format(data)
print(f"\nConverted: {converted.shape}, columns: {converted.columns.tolist()}")
print(f"signal[0] shape: {converted['signal'].iloc[0].shape}, response_time sample: {converted['response_time'].head(3).tolist()}")
