"""
Main evaluation pipeline for pre-trained models

This module orchestrates the evaluation of all pre-trained models on R2_mini_L100_bdf test data,
computing normalized RMSE and other metrics for each behavioral target.
"""

import argparse
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.ML_pipeline_test import config
from src.ML_pipeline_test.models import CNN1DAutoencoder
from src.ML_pipeline_test.regression import RegressionHead

from .eval_data_prep import create_evaluation_datasets, verify_evaluation_datasets
from .metrics import calculate_all_metrics, format_metrics_for_display

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)


def load_pretrained_encoder(model_path: Path = None) -> torch.nn.Module:
    """
    Load pretrained encoder from autoencoder checkpoint
    
    Args:
        model_path: Path to autoencoder checkpoint
        
    Returns:
        Encoder module from autoencoder
    """
    if model_path is None:
        model_path = config.AUTOENCODER_PATH

    if not model_path.exists():
        raise FileNotFoundError(f"Autoencoder not found at {model_path}")

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=config.DEVICE)

    # Create autoencoder and load weights
    autoencoder = CNN1DAutoencoder()
    autoencoder.load_state_dict(checkpoint['model_state_dict'])

    print(f"Loaded encoder from epoch {checkpoint.get('epoch', 'unknown')}")

    # Return just the encoder part
    encoder = autoencoder.encoder
    encoder.eval()  # Set to evaluation mode

    return encoder


def load_regression_model(model_path: Path, encoder: torch.nn.Module) -> RegressionHead:
    """
    Load a pre-trained regression model
    
    Args:
        model_path: Path to regression model checkpoint
        encoder: Pre-trained encoder module
        
    Returns:
        Loaded regression model
    """
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")

    # Create model with pretrained encoder
    model = RegressionHead(encoder=encoder, freeze_encoder=True)

    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=config.DEVICE)

    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model.to(config.DEVICE)
    model.eval()  # Set to evaluation mode

    return model


def evaluate_single_model(
    model_path: Path,
    encoder: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    target_name: str,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate a single regression model on test data
    
    Args:
        model_path: Path to model checkpoint
        encoder: Pre-trained encoder
        dataloader: DataLoader for test data
        target_name: Name of the target being predicted
        verbose: Whether to print progress
        
    Returns:
        Dictionary containing evaluation results
    """
    try:
        # Load model
        model = load_regression_model(model_path, encoder)

        if verbose:
            print(f"\nEvaluating {target_name} model...")

        # Collect predictions and targets
        all_predictions = []
        all_targets = []

        with torch.no_grad():
            for batch_data, batch_targets in tqdm(dataloader, desc=f"Evaluating {target_name}", disable=not verbose):
                # Move to device
                batch_data = batch_data.to(config.DEVICE)
                batch_targets = batch_targets.to(config.DEVICE)

                # Get predictions
                predictions = model(batch_data)

                # Store results
                all_predictions.extend(predictions.cpu().numpy())
                all_targets.extend(batch_targets.cpu().numpy())

        # Convert to numpy arrays
        y_true = np.array(all_targets)
        y_pred = np.array(all_predictions)

        # Calculate metrics
        metrics = calculate_all_metrics(y_true, y_pred)

        # Add metadata
        results = {
            'target_name': target_name,
            'model_path': str(model_path),
            'num_samples': len(y_true),
            'metrics': metrics,
            'predictions': y_pred,
            'targets': y_true
        }

        if verbose:
            print(format_metrics_for_display(metrics))

        return results

    except Exception as e:
        print(f"Error evaluating {target_name}: {e}")
        return {
            'target_name': target_name,
            'model_path': str(model_path),
            'error': str(e),
            'metrics': None
        }


def create_evaluation_dataloader(
    dataset_path: Path,
    target_column: str,
    batch_size: int = 32,
    participants_tsv_path: Path = None
) -> torch.utils.data.DataLoader:
    """
    Create DataLoader for evaluation dataset
    
    Args:
        dataset_path: Path to pickled dataset
        target_column: Column name for target values
        batch_size: Batch size for DataLoader
        participants_tsv_path: Path to participants.tsv file for behavioral factors
        
    Returns:
        DataLoader for evaluation
    """
    # Load dataset
    with open(dataset_path, 'rb') as f:
        df = pickle.load(f)
    
    # Check if target is a psychopathology factor
    psychopathology_factors = ["p_factor", "attention", "internalizing", "externalizing"]
    is_psychopathology_target = target_column in psychopathology_factors
    
    if is_psychopathology_target:
        if participants_tsv_path is None:
            raise ValueError(f"participants_tsv_path is required for psychopathology factor '{target_column}'")
        
        # Load participants data for psychopathology factor lookup
        participants_df = pd.read_csv(participants_tsv_path, sep='\t')
        
        # Create mapping from subject_id to factor value
        psycho_factor_map = {}
        for _, row in participants_df.iterrows():
            # Handle both with and without 'sub-' prefix
            subject_id = row['participant_id'].replace('sub-', '')
            psycho_factor_map[subject_id] = row[target_column]
        
        print(f"Loaded psychopathology factor '{target_column}' for {len(psycho_factor_map)} participants")
        
        # Extract signals and labels with behavioral factor lookup
        signals = []
        labels = []
        
        for _, row in df.iterrows():
            signal = row['signal']
            subject_id = row['subject']
            
            # Get behavioral factor value for this subject
            if subject_id in psycho_factor_map:
                target_value = psycho_factor_map[subject_id]
                # Skip if behavioral factor is NaN
                if not pd.isna(target_value):
                    signals.append(signal)
                    labels.append(target_value)
        
        if len(signals) == 0:
            raise ValueError(f"No valid samples found with behavioral factor '{target_column}'")
        
        signals = np.array(signals)
        labels = np.array(labels)
        
    else:
        # Handle non-psychopathology targets (like response_time)
        # Filter out samples with missing targets
        df = df[df[target_column].notna()]
        
        if len(df) == 0:
            raise ValueError(f"No valid samples found for target '{target_column}'")
        
        signals = np.stack(df['signal'].values)
        labels = df[target_column].values
    
    # Create torch dataset
    class EvalDataset(torch.utils.data.Dataset):
        def __init__(self, signals, labels):
            self.signals = torch.FloatTensor(signals)
            self.targets = torch.FloatTensor(labels)

        def __len__(self):
            return len(self.signals)

        def __getitem__(self, idx):
            return self.signals[idx], self.targets[idx]

    dataset = EvalDataset(signals, labels)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    print(f"Created dataloader with {len(dataset)} samples for {target_column}")

    return dataloader


def run_comprehensive_evaluation(
    create_datasets: bool = True,
    output_dir: Path = None,
    dataset_dir: Path = None,
    test_mode: bool = False,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run evaluation on all available models
    
    Args:
        create_datasets: Whether to create evaluation datasets
        output_dir: Directory for saving results
        dataset_dir: Directory containing evaluation datasets
        test_mode: Run in test mode with reduced data
        verbose: Whether to print progress
        
    Returns:
        DataFrame with evaluation results
    """
    if output_dir is None:
        output_dir = PROJECT_ROOT / "src" / "ML_pipeline_test" / "saved_models"
        if not output_dir.exists():
            output_dir.mkdir(parents=True, exist_ok=True)
    
    if dataset_dir is None:
        dataset_dir = PROJECT_ROOT / "datasets" / "evaluation_datasets"
    
    # Set participants.tsv path for R2_mini_L100_bdf evaluation database
    participants_tsv_path = PROJECT_ROOT / "database" / "R2_mini_L100_bdf" / "participants.tsv"
    print(f"WARNING; check that {participants_tsv_path} is correct (R2 mini L100 bdf is probably renamed)")
    exit()

    print("=" * 60)
    print("COMPREHENSIVE MODEL EVALUATION")
    print("=" * 60)

    # Step 1: Create or verify evaluation datasets
    if create_datasets:
        print("\n[1/4] Creating evaluation datasets...")
        subjects_limit = 5 if test_mode else None
        created_files = create_evaluation_datasets(
            subjects_limit=subjects_limit,
            verbose=verbose
        )
    else:
        print("\n[1/4] Verifying existing datasets...")
        if not verify_evaluation_datasets(dataset_dir):
            raise RuntimeError("Evaluation datasets not found. Run with --create-datasets flag.")

    # Step 2: Load pre-trained encoder
    print("\n[2/4] Loading pre-trained encoder...")
    encoder = load_pretrained_encoder()

    # Step 3: Define model configurations
    root_eval_datapath = dataset_dir
    model_configs = [
        {
            'name': 'attention',
            'model_path': config.MODEL_DIR / 'regressor_attention_best.pth',
            'dataset_path': root_eval_datapath / 'pretraining_data_shard_0.pkl',
            'target_column': 'attention'
        },
        {
            'name': 'externalizing',
            'model_path': config.MODEL_DIR / 'regressor_externalizing_best.pth',
            'dataset_path': root_eval_datapath / 'pretraining_data_shard_0.pkl',
            'target_column': 'externalizing'
        },
        {
            'name': 'internalizing',
            'model_path': config.MODEL_DIR / 'regressor_internalizing_best.pth',
            'dataset_path': root_eval_datapath / 'pretraining_data_shard_0.pkl',
            'target_column': 'internalizing'
        },
        {
            'name': 'p_factor',
            'model_path': config.MODEL_DIR / 'regressor_p_factor_best.pth',
            'dataset_path': root_eval_datapath / 'pretraining_data_shard_0.pkl',
            'target_column': 'p_factor'
        },
        {
            'name': 'response_time',
            'model_path': config.MODEL_DIR / 'regressor_response_time_best.pth',
            'dataset_path': root_eval_datapath / 'challenge1_data_shard_0.pkl',
            'target_column': 'response_time'
        }
    ]

    # Step 4: Evaluate each model
    print("\n[3/4] Evaluating models...")
    all_results = []

    for config_dict in model_configs:
        model_name = config_dict['name']
        model_path = config_dict['model_path']
        dataset_path = config_dict['dataset_path']
        target_column = config_dict['target_column']

        print(f"\n{'='*40}")
        print(f"Evaluating: {model_name}")
        print(f"{'='*40}")

        # Check if model exists
        if not model_path.exists():
            print(f"⚠ Model not found: {model_path}")
            all_results.append({
                'model': model_name,
                'status': 'not_found',
                'nrmse': None,
                'rmse': None,
                'mae': None,
                'r2': None,
                'num_samples': 0
            })
            continue

        # Check if dataset exists
        if not dataset_path.exists():
            print(f"⚠ Dataset not found: {dataset_path}")
            all_results.append({
                'model': model_name,
                'status': 'no_data',
                'nrmse': None,
                'rmse': None,
                'mae': None,
                'r2': None,
                'num_samples': 0
            })
            continue

        try:
            # Create dataloader
            dataloader = create_evaluation_dataloader(
                dataset_path,
                target_column,
                batch_size=32 if not test_mode else 8,
                participants_tsv_path=participants_tsv_path
            )

            # Evaluate model
            results = evaluate_single_model(
                model_path,
                encoder,
                dataloader,
                model_name,
                verbose=verbose
            )

            # Extract metrics
            if results['metrics'] is not None:
                metrics = results['metrics']
                all_results.append({
                    'model': model_name,
                    'status': 'success',
                    'nrmse': metrics['nrmse'],
                    'rmse': metrics['rmse'],
                    'mae': metrics['mae'],
                    'r2': metrics['r2'],
                    'correlation': metrics.get('correlation', None),
                    'num_samples': results['num_samples']
                })
            else:
                all_results.append({
                    'model': model_name,
                    'status': 'error',
                    'error': results.get('error', 'Unknown error'),
                    'nrmse': None,
                    'rmse': None,
                    'mae': None,
                    'r2': None,
                    'num_samples': 0
                })

        except Exception as e:
            print(f"Error evaluating {model_name}: {e}")
            all_results.append({
                'model': model_name,
                'status': 'error',
                'error': str(e),
                'nrmse': None,
                'rmse': None,
                'mae': None,
                'r2': None,
                'num_samples': 0
            })

    # Step 5: Save results
    print("\n[4/4] Saving results...")
    results_df = pd.DataFrame(all_results)

    # Save to CSV
    csv_path = output_dir / 'evaluation_results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"Results saved to: {csv_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(results_df.to_string())

    # Print best model for primary metric (NRMSE)
    valid_results = results_df[results_df['nrmse'].notna()]
    if not valid_results.empty:
        best_model = valid_results.loc[valid_results['nrmse'].idxmin()]
        print("\n" + "=" * 60)
        print(f"BEST MODEL (lowest NRMSE): {best_model['model']}")
        print(f"  NRMSE: {best_model['nrmse']:.4f}")
        print(f"  R²: {best_model['r2']:.4f}")
        print("=" * 60)

    return results_df


def main():
    """Main entry point for evaluation script"""
    parser = argparse.ArgumentParser(description="Evaluate pre-trained models on R2_mini_L100_bdf data")
    parser.add_argument("--create-datasets", action="store_true", help="Create evaluation datasets")
    parser.add_argument("--test-mode", action="store_true", help="Run in test mode with limited data")
    parser.add_argument("--output-dir", type=str, help="Output directory for results")
    parser.add_argument("--dataset-dir", type=str, help="Directory containing evaluation datasets")
    parser.add_argument("--single-model", type=str, help="Evaluate only a single model (e.g., 'attention')")
    parser.add_argument("--verbose", action="store_true", default=True, help="Print detailed progress")

    args = parser.parse_args()

    if args.single_model:
        # Evaluate single model
        print(f"Evaluating single model: {args.single_model}")
        # Implementation for single model evaluation
    else:
        # Run comprehensive evaluation
        run_comprehensive_evaluation(
            create_datasets=args.create_datasets,
            output_dir=Path(args.output_dir) if args.output_dir else None,
            dataset_dir=Path(args.dataset_dir) if args.dataset_dir else None,
            test_mode=args.test_mode,
            verbose=args.verbose
        )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
