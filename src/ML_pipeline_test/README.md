# PyTorch EEG ML Pipeline

Minimal PyTorch pipeline for EEG pretraining using masked autoencoding and downstream classification of button press events.

## Overview

This implementation provides a complete ML pipeline for EEG data analysis with:
- Modular architecture with clean separation of concerns
- Masked autoencoder pretraining on 968 EEG epochs
- Flexible downstream tasks (classification and regression)
- Config-driven task switching with auto-generated label mappings

## Project Structure

```
ML_pipeline_test/
├── config.py           # Configuration parameters (64 lines)
├── data_loader_ml.py   # Data loading (Self-supervised and DownstreamTask) (172 lines)
├── models.py          # Neural network architectures (171 lines)
├── pretraining.py     # Masked autoencoder training (160 lines)
├── classification.py   # Downstream classification (196 lines)
├── utils.py           # Helper functions and evaluation (183 lines)
├── run_pipeline.sh    # Complete pipeline execution script
└── saved_models/      # Trained model checkpoints
    ├── autoencoder_best.pth
    └── classifier_best.pth
```

## Data Specifications

### Pretraining Data
- **File**: `data/pretraining_data.pkl`

### Posttraining Data
- **File**: `data/posttraining_data.pkl`

### Challenge_1 Data
- **File**: `data/challenge_1_data.pkl`

## Model Architecture

### CNN1DAutoencoder
- **Encoder**: 3 Conv1d layers (129→64→32→16 channels)
- **Decoder**: Mirror architecture with ConvTranspose1d
- **Masking**: Random 75% time segment masking
- **Kernel Size**: 5 for all convolution layers

### BinaryClassifier
- **Backbone**: Pretrained encoder (frozen by default)
- **Head**: Global average pooling → Linear(16→32) → Linear(32→2)
- **Dropout**: 0.3 for regularization

## Training Configuration

### Autoencoder Pretraining
- **Epochs**: 100
- **Batch Size**: 32
- **Learning Rate**: 1e-3
- **Loss**: MSE on reconstruction
- **Optimizer**: Adam with weight decay 1e-5

### Classification Training
- **Epochs**: 50
- **Batch Size**: 16
- **Learning Rate**: 1e-3
- **Loss**: CrossEntropy
- **Optimizer**: Adam with weight decay 1e-5

## Usage

### Quick Start
```bash
# Run complete pipeline
bash ML_pipeline_test/run_pipeline.sh
```

### Individual Components

```bash
# Activate environment
source venv_HBN_linux/bin/activate

# Check data
python ML_pipeline_test/utils.py --check-data

# Train autoencoder
python ML_pipeline_test/pretraining.py --epochs 100 --batch-size 32

# Train classifier (button press or hit accuracy)
python ML_pipeline_test/classification.py --epochs 50 --batch-size 16

# Train regressor (response time # challenge 1)
python ML_pipeline_test/regression.py --epochs 50 --batch-size 16 --target response_time

# Train regressor (p_factor # challenge 2; works with attention, internalizing, externalizing)
python ML_pipeline_test/regression.py --epochs 50 --batch-size 16 --target p_factor

# Evaluate models
python ML_pipeline_test/utils.py --evaluate --model-type autoencoder
python ML_pipeline_test/utils.py --evaluate --model-type classifier
```

### Custom Training

```python
from data_loader_ml import create_dataloaders
from models import CNN1DAutoencoder, BinaryClassifier

# Load data
train_loader, val_loader = create_dataloaders('pretraining', batch_size=32)

# Initialize model
model = CNN1DAutoencoder().to('cuda')

# Train with custom settings
from pretraining import train_autoencoder
model, train_losses, val_losses = train_autoencoder(epochs=200, batch_size=64)
```

## Current Performance

### Autoencoder (5 epochs test)
- **Training Loss**: 0.0000
- **Validation Loss**: 0.0000
- **Reconstruction MSE**: < 0.0001

### Classifier (5 epochs test)
- **Training Accuracy**: 56.6%
- **Validation Accuracy**: 44.4%

## Implementation Status

### Completed ✓
- Data loading and preprocessing pipelines
- 3-layer 1D CNN autoencoder with masking
- Binary classifier with pretrained encoder
- Training loops with validation
- Model checkpointing and evaluation
- All scripts under 200 lines requirement


## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- Pandas
- scikit-learn
- tqdm

## Future Improvements

2. **Advanced Masking**: Implement channel masking, variable ratios
3. **Architecture**: Experiment with attention mechanisms
4. **Multi-task**: Extend to multi-class classification
5. **Hyperparameter Tuning**: Grid search for optimal parameters
