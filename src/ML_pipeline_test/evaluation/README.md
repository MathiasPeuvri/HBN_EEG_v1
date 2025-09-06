# Evaluation Module for HBN EEG Analysis

This module provides comprehensive evaluation capabilities for pre-trained EEG models, computing normalized RMSE and other metrics on test data.

## Features

- **Normalized RMSE Calculation**: Primary evaluation metric using standard deviation normalization
- **Comprehensive Metrics**: RMSE, MAE, R², correlation coefficient, and residual analysis
- **Multi-Model Support**: Evaluates behavioral (attention, externalizing, internalizing, p-factor) and response time models
- **Flexible Data Loading**: Supports both real and synthetic evaluation datasets
- **CSV Export**: Results automatically saved for analysis and comparison

## Installation

The module is part of the HBN_EEG_Analysis project. No additional installation required beyond the main project dependencies.

## Usage

### Command Line Interface

```bash
# Run evaluation with dataset creation
python -m src.ML_pipeline_test.evaluation.evaluation --create-datasets

# Run in test mode (limited data)
python -m src.ML_pipeline_test.evaluation.evaluation --test-mode

# Specify output directory
python -m src.ML_pipeline_test.evaluation.evaluation --output-dir /path/to/results

# Evaluate single model
python -m src.ML_pipeline_test.evaluation.evaluation --single-model attention
```

### Python API

```python
from src.ML_pipeline_test.evaluation import run_comprehensive_evaluation

# Run full evaluation
results_df = run_comprehensive_evaluation(
    create_datasets=True,
    test_mode=False,
    verbose=True
)

# Access results
print(results_df[['model', 'nrmse', 'r2']])
```

### Metrics Calculation

```python
from src.ML_pipeline_test.evaluation.metrics import calculate_all_metrics

# Calculate metrics for predictions
metrics = calculate_all_metrics(y_true, y_pred)
print(f"NRMSE: {metrics['nrmse']:.4f}")
print(f"R² Score: {metrics['r2']:.4f}")
```

## Module Structure

```
src/ML_pipeline_test/evaluation/
├── __init__.py              # Module exports
├── evaluation.py            # Main evaluation pipeline
├── metrics.py              # Evaluation metrics
├── eval_data_prep.py       # Dataset preparation
└── README.md               # This file
```

## Evaluation Metrics

### Primary Metric
- **NRMSE (Normalized RMSE)**: RMSE divided by standard deviation of true values
  - Range: 0 to ∞ (lower is better)
  - < 1.0 indicates predictions better than mean baseline

### Secondary Metrics
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error  
- **R²**: Coefficient of determination
- **Correlation**: Pearson correlation coefficient

## Dataset Requirements

### Expected Format
Datasets should be pickled pandas DataFrames with:
- `signal`: numpy array of shape (129, 200) - EEG signal data
- Target columns: `p_factor`, `attention`, `internalizing`, `externalizing`, or `response_time`
- `subject`: Subject identifier (optional)

### File Locations
- Posttraining data: `dataset/evaluation_datasets/posttraining_eval_data.pkl`
- Challenge1 data: `dataset/evaluation_datasets/challenge1_eval_data.pkl`
- Results CSV: `dataset/evaluation_datasets/evaluation_results.csv`

## Model Requirements

Pre-trained models should be saved in `src/ML_pipeline_test/saved_models/`:
- `autoencoder_best.pth`: Feature extraction encoder
- `regressor_attention_best.pth`: Attention prediction model
- `regressor_externalizing_best.pth`: Externalizing behavior model
- `regressor_internalizing_best.pth`: Internalizing behavior model
- `regressor_p_factor_best.pth`: P-factor prediction model
- `regressor_response_time_best.pth`: Response time prediction model

## Testing

Run unit tests:
```bash
python -m pytest tests/test_evaluation.py -v
```

## Known Limitations

- Currently requires .pkl format for datasets
- R2_mini_L100_bdf database requires BDF file support (pending implementation)
- Models must follow specific checkpoint structure

## Contributing

When extending this module:
1. Add new metrics to `metrics.py`
2. Update `calculate_all_metrics()` to include new metrics
3. Add corresponding unit tests
4. Update this documentation

## References

- [Normalized RMSE Guide](https://www.marinedatascience.co/blog/2019/01/07/normalizing-the-rmse/)
- [PyTorch Model Loading](https://docs.pytorch.org/tutorials/beginner/saving_loading_models.html)
- [Scikit-learn Metrics](https://scikit-learn.org/stable/modules/model_evaluation.html)