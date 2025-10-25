#!/usr/bin/env python3
"""
Script to analyze predictions.pickle file from the EEG challenge submission
"""

import pickle
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_predictions(pickle_path):
    """Load predictions from pickle file"""
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    return data


def analyze_predictions(predictions):
    """Analyze the structure and content of predictions"""
    print("=" * 80)
    print("PREDICTIONS FILE ANALYSIS")
    print("=" * 80)

    # Type and structure
    print(f"\nType: {type(predictions)}")

    if isinstance(predictions, dict):
        print(f"\nKeys: {list(predictions.keys())}")
        print("\n" + "-" * 80)

        for key, value in predictions.items():
            print(f"\nKey: '{key}'")
            print(f"  Type: {type(value)}")

            if isinstance(value, (list, np.ndarray)):
                arr = np.array(value)
                print(f"  Shape: {arr.shape}")
                print(f"  Dtype: {arr.dtype}")
                print(f"  Min: {np.min(arr):.4f}")
                print(f"  Max: {np.max(arr):.4f}")
                print(f"  Mean: {np.mean(arr):.4f}")
                print(f"  Std: {np.std(arr):.4f}")
                print(f"  First 5 values: {arr.flatten()[:5]}")

                # Check for NaN or Inf
                if np.isnan(arr).any():
                    print(f"  Contains NaN values: {np.isnan(arr).sum()}")
                if np.isinf(arr).any():
                    print(f"  Contains Inf values: {np.isinf(arr).sum()}")

            elif isinstance(value, dict):
                print(f"  Sub-keys: {list(value.keys())}")
                for sub_key, sub_value in value.items():
                    print(f"    '{sub_key}': {type(sub_value)}", end="")
                    if isinstance(sub_value, (list, np.ndarray)):
                        arr = np.array(sub_value)
                        print(f" - Shape: {arr.shape}")
                    else:
                        print()
            else:
                print(f"  Value: {value}")

    elif isinstance(predictions, (list, np.ndarray)):
        arr = np.array(predictions)
        print(f"\nShape: {arr.shape}")
        print(f"Dtype: {arr.dtype}")
        print(f"Min: {np.min(arr):.4f}")
        print(f"Max: {np.max(arr):.4f}")
        print(f"Mean: {np.mean(arr):.4f}")
        print(f"Std: {np.std(arr):.4f}")
        print(f"\nFirst 10 values:\n{arr.flatten()[:10]}")

    else:
        print(f"\nContent: {predictions}")

    print("\n" + "=" * 80)


def main():
    # Path to the predictions file
    predictions_path = Path("test_submission_output/predictions.pickle")

    if not predictions_path.exists():
        print(f" File not found: {predictions_path}")
        return

    print(f"Loading predictions from: {predictions_path}\n")

    # Load and analyze
    predictions = load_predictions(predictions_path)
    analyze_predictions(predictions)

    # Plot distributions if predictions is a dict with challenges
    if isinstance(predictions, dict) and 'challenge_1' in predictions:
        plot_distributions(predictions)


def plot_distributions(predictions):
    """Plot distribution of predictions vs true values for each challenge"""
    print("\n" + "=" * 80)
    print("GENERATING DISTRIBUTION PLOTS")
    print("=" * 80)

    challenges = {'challenge_1': 'Reaction Time', 'challenge_2': 'Externalizing'}

    fig, axes = plt.subplots(1, 2, figsize=(15, 10))
    fig.suptitle('Predictions vs True Values Distribution', fontsize=16, fontweight='bold')

    for idx, (challenge_key, challenge_name) in enumerate(challenges.items()):
        if challenge_key not in predictions:
            continue

        y_preds = predictions[challenge_key]['y_preds']
        y_trues = predictions[challenge_key]['y_trues']

        # Distribution plots
        ax_dist = axes[idx]
        ax_dist.hist(y_trues, bins=50, alpha=0.6, label='True values', color='blue', density=False)
        ax_dist.hist(y_preds, bins=50, alpha=0.6, label='Predictions', color='red', density=False)
        ax_dist.set_xlabel('Value')
        ax_dist.set_ylabel('Density')
        ax_dist.set_title(f'{challenge_name} - Distribution')
        ax_dist.legend()
        ax_dist.grid(True, alpha=0.3)

        # Print statistics
        print(f"\n{challenge_name}:")
        print(f"  True values  - Min: {y_trues.min():.4f}, Max: {y_trues.max():.4f}, Mean: {y_trues.mean():.4f}, Std: {y_trues.std():.4f}")
        print(f"  Predictions  - Min: {y_preds.min():.4f}, Max: {y_preds.max():.4f}, Mean: {y_preds.mean():.4f}, Std: {y_preds.std():.4f}")
        print(f"  N samples: {len(y_preds)}")

    plt.tight_layout()
    output_path = Path('test_submission_output/predictions_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
