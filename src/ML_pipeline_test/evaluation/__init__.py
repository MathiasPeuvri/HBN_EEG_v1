"""
Evaluation module for HBN EEG Analysis

This module provides functionality to evaluate pre-trained models on unseen R2_mini_L100_bdf data,
computing normalized RMSE and other metrics for behavioral prediction tasks.
"""

from .eval_data_prep import create_evaluation_datasets, verify_evaluation_datasets
from .evaluation import evaluate_single_model, run_comprehensive_evaluation
from .metrics import calculate_all_metrics, calculate_normalized_rmse

__all__ = [
    'run_comprehensive_evaluation',
    'evaluate_single_model',
    'calculate_normalized_rmse',
    'calculate_all_metrics',
    'create_evaluation_datasets',
    'verify_evaluation_datasets'
]
