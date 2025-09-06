"""
Evaluation metrics for regression models

This module provides functions to calculate various evaluation metrics
for regression tasks, with normalized RMSE as the primary metric.
"""

from typing import Dict, Tuple

import numpy as np
import scipy.stats as stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate_normalized_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate normalized RMSE using standard deviation normalization
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        float: NRMSE value (RMSE / std(y_true))
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Input arrays cannot be empty")

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    std_true = np.std(y_true)

    if std_true == 0:
        raise ValueError("Standard deviation of y_true is zero")

    return rmse / std_true


def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive set of regression metrics
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        dict: Dictionary containing multiple metrics
    """
    metrics = {
        'nrmse': calculate_normalized_rmse(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mean_target': np.mean(y_true),
        'std_target': np.std(y_true),
        'mean_pred': np.mean(y_pred),
        'std_pred': np.std(y_pred)
    }

    # Add correlation coefficient
    correlation = np.corrcoef(y_true, y_pred)[0, 1]
    metrics['correlation'] = correlation

    return metrics


def calculate_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Calculate residuals (errors) between predictions and true values
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        np.ndarray: Array of residuals (y_true - y_pred)
    """
    return y_true - y_pred


def analyze_residuals(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Perform residual analysis for error diagnostics
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        dict: Dictionary containing residual statistics
    """
    residuals = calculate_residuals(y_true, y_pred)

    residual_stats = {
        'mean_residual': np.mean(residuals),
        'std_residual': np.std(residuals),
        'min_residual': np.min(residuals),
        'max_residual': np.max(residuals),
        'median_residual': np.median(residuals),
        'q25_residual': np.percentile(residuals, 25),
        'q75_residual': np.percentile(residuals, 75)
    }

    # Test for normality of residuals (Shapiro-Wilk test)
    if len(residuals) >= 3:
        statistic, p_value = stats.shapiro(residuals)
        residual_stats['shapiro_statistic'] = statistic
        residual_stats['shapiro_pvalue'] = p_value

    return residual_stats


def calculate_confidence_intervals(y_true: np.ndarray, y_pred: np.ndarray,
                                 confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calculate confidence intervals for predictions
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        confidence: Confidence level (default 0.95 for 95% CI)
        
    Returns:
        tuple: (lower_bound, upper_bound) of confidence interval for RMSE
    """
    n = len(y_true)
    residuals = calculate_residuals(y_true, y_pred)

    # Standard error of residuals
    se = np.std(residuals) / np.sqrt(n)

    # Calculate confidence interval using t-distribution
    alpha = 1 - confidence
    t_score = stats.t.ppf(1 - alpha/2, df=n-1)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    ci_lower = rmse - t_score * se
    ci_upper = rmse + t_score * se

    return ci_lower, ci_upper


def calculate_relative_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate mean relative error (percentage)
    
    Args:
        y_true: Array of true values
        y_pred: Array of predicted values
        
    Returns:
        float: Mean relative error as percentage
    """
    # Avoid division by zero
    mask = y_true != 0
    if not np.any(mask):
        return np.inf

    relative_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return np.mean(relative_errors) * 100


def format_metrics_for_display(metrics: Dict[str, float]) -> str:
    """
    Format metrics dictionary for console display
    
    Args:
        metrics: Dictionary of metric names and values
        
    Returns:
        str: Formatted string for display
    """
    output = []
    output.append("=" * 50)
    output.append("Evaluation Metrics")
    output.append("=" * 50)

    # Primary metrics
    if 'nrmse' in metrics:
        output.append(f"NRMSE: {metrics['nrmse']:.4f}")
    if 'rmse' in metrics:
        output.append(f"RMSE: {metrics['rmse']:.4f}")
    if 'mae' in metrics:
        output.append(f"MAE: {metrics['mae']:.4f}")
    if 'r2' in metrics:
        output.append(f"R² Score: {metrics['r2']:.4f}")
    if 'correlation' in metrics:
        output.append(f"Correlation: {metrics['correlation']:.4f}")

    # Target statistics
    output.append("-" * 50)
    if 'mean_target' in metrics:
        output.append(f"Target Mean: {metrics['mean_target']:.4f}")
    if 'std_target' in metrics:
        output.append(f"Target Std: {metrics['std_target']:.4f}")
    if 'mean_pred' in metrics:
        output.append(f"Prediction Mean: {metrics['mean_pred']:.4f}")
    if 'std_pred' in metrics:
        output.append(f"Prediction Std: {metrics['std_pred']:.4f}")

    output.append("=" * 50)

    return "\n".join(output)
