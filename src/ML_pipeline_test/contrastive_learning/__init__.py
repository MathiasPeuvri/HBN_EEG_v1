"""
Contrastive Representation Learning (CRL) Module for HBN EEG Analysis

Implementation adapted from Mohsenvand et al. (2020):
"Contrastive Representation Learning for Electroencephalogram Classification"

This module implements contrastive learning for EEG pretraining using:
- 6 data augmentation strategies
- NT-Xent loss function
- Convolutional encoder with bi-LSTM projector
- Multi-task pretraining on HBN dataset (6 tasks)

Adapted for HBN dataset:
- 129 channels (128 EEG + 1 reference)
- 100 Hz sampling rate
- 2-second windows (200 samples)
"""

from .config import CRL_CONFIG
from .augmentations import apply_augmentations, transformation_ranges
from .dataset import ContrastiveEEGDataset
from .models import ConvolutionalEncoder, Projector, EEGContrastiveModel
from .loss import NTXentLoss
from .trainer import pretrain_contrastive

__all__ = [
    "CRL_CONFIG",
    "apply_augmentations",
    "transformation_ranges",
    "ContrastiveEEGDataset",
    "ConvolutionalEncoder",
    "Projector",
    "EEGContrastiveModel",
    "NTXentLoss",
    "pretrain_contrastive",
]
