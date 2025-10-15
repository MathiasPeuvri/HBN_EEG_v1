"""
Neural Network Models for Contrastive Learning on EEG

Implements the architecture from Mohsenvand et al. (2020):
- ConvolutionalEncoder: Multi-branch CNN with residual connections
- Projector: Multi-scale bi-LSTM with FLO (First-Last-Output) strategy
- EEGContrastiveModel: Combined encoder + projector for end-to-end training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))
from src.preprocessing.preprocessor import EEGPreprocessor
from .config import N_CHANS, SAMPLEPOINTS, PROJECTOR_OUTPUT_DIM


class ConvolutionalEncoder(nn.Module):
    """
    Multi-branch convolutional encoder for EEG signals.

    Architecture:
    - 3 parallel branches with different kernel sizes (multi-scale feature extraction)
    - Reflection padding to maintain temporal dimensions
    - 4 residual blocks with batch normalization
    - Adaptive kernel scaling based on input length

    Args:
        in_channels: Number of input channels (default: 129 for HBN)
        repeat_n: Number of residual blocks (default: 4)
        n_samples: Number of time samples per epoch (default: 200 for 2s at 100Hz)
    """

    def __init__(self, in_channels: int = N_CHANS, repeat_n: int = 4, n_samples: int = SAMPLEPOINTS):
        super(ConvolutionalEncoder, self).__init__()

        # Scale kernel sizes based on signal length
        # Original paper: 4000 samples with kernels [128, 64, 16]
        # Maintain proportional receptive fields for shorter signals
        scale_factor = n_samples / 4000.0
        k1 = max(4, int(128 * scale_factor))  # ~6 for 200 samples 
        k2 = max(4, int(64 * scale_factor))   # ~3 for 200 samples
        k3 = max(2, int(16 * scale_factor))   # ~1 for 200 samples
        # potentiellement regarder pour permutter x sur une des branches pour faire une conv spatial

        # # proposition maxim # beaucoup d'alpha, correler taille des filtres de conv avec ce qu'on veut voir niveau freq
        # k1 = max(4, int(128 * scale_factor))  # 20 -> voir l'alpha
        # k2 = max(4, int(64 * scale_factor))   # 10 -> voir l'alpha
        # k3 = max(2, int(16 * scale_factor))   # 5 -> voir le delta / mouvement yeux / perturbations etc ...
        # plutot que couper un branche, duppliquer les branches d'entré pour prendre les sources indépendament

        self.k1, self.k2, self.k3 = k1, k2, k3

        # Three parallel branches with different kernel sizes
        self.branch1 = nn.Conv1d(in_channels, 100, kernel_size=k1, padding=0)
        self.branch2 = nn.Conv1d(in_channels, 100, kernel_size=k2, padding=0)
        self.branch3 = nn.Conv1d(in_channels, 50, kernel_size=k3, padding=0)

        # Dense layer after concatenation (100+100+50=250)
        self.dense1 = nn.Linear(250, 250)

        # Repeated residual blocks (N=4)
        self.repeat_blocks = nn.ModuleList()
        for _ in range(repeat_n):
            block = nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm1d(250),
                nn.Conv1d(250, 250, kernel_size=64, padding=0)
            )
            self.repeat_blocks.append(block)

        # Final layers
        self.final_relu = nn.ReLU()
        self.final_bn = nn.BatchNorm1d(250)
        self.final_conv = nn.Conv1d(250, 4, kernel_size=64, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.

        Args:
            x: Input tensor (batch, channels, time)

        Returns:
            Encoded features (batch, 4, time_reduced)
        """
        # potentiellement regarder pour permutter x sur une des branches pour faire une conv spatial -> no padding?
        # par ici qu'il faut revoir, quand on fait rentrer les donnés

        # Apply reflection padding for each branch (maintains temporal dimension)
        pad1_left = self.k1 // 2
        pad1_right = self.k1 - pad1_left - 1
        x1 = F.pad(x, (pad1_left, pad1_right), mode='reflect')
        x1 = self.branch1(x1)

        pad2_left = self.k2 // 2
        pad2_right = self.k2 - pad2_left - 1
        x2 = F.pad(x, (pad2_left, pad2_right), mode='reflect')
        x2 = self.branch2(x2)

        pad3_left = self.k3 // 2
        pad3_right = self.k3 - pad3_left - 1
        x3 = F.pad(x, (pad3_left, pad3_right), mode='reflect')
        x3 = self.branch3(x3)

        # Concatenate along channel dimension
        x = torch.cat([x1, x2, x3], dim=1)  # (batch, 250, time)

        # Dense layer (permute for Linear operation)
        x = x.permute(0, 2, 1)  # (batch, time, 250)
        x = self.dense1(x)
        x = x.permute(0, 2, 1)  # (batch, 250, time)

        # Residual blocks
        for block in self.repeat_blocks:
            residual = x
            x = F.pad(x, (32, 31), mode='reflect')  # Padding for kernel=64
            x = block(x)
            # Adjust size if necessary for residual addition
            if x.shape[2] != residual.shape[2]:
                min_len = min(x.shape[2], residual.shape[2])
                x = x[:, :, :min_len]
                residual = residual[:, :, :min_len]
            x = x + residual

        # Final layers
        x = self.final_relu(x)
        x = self.final_bn(x)
        x = F.pad(x, (32, 31), mode='reflect')
        x = self.final_conv(x)

        return x


class Projector(nn.Module):
    """
    Multi-scale bi-LSTM projector with FLO (First-Last-Output) strategy.

    Projects encoder outputs into a lower-dimensional space for contrastive learning.
    Can also be used for supervised tasks by setting task_mode appropriately.

    Architecture:
    - 3 bi-LSTM branches (different downsampling factors)
    - FLO strategy (concatenate first and last timestep outputs)
    - 2-layer MLP for final projection

    Args:
        input_dim: Input feature dimension (encoder output channels, default: 4)
        output_dim: Output projection dimension (default: 128 for contrastive learning)
        task_mode: Task type - 'contrastive', 'classification', or 'regression' (default: 'contrastive')
    """

    def __init__(
        self,
        input_dim: int = 4,
        output_dim: int = PROJECTOR_OUTPUT_DIM,
        task_mode: str = 'contrastive'
    ):
        super(Projector, self).__init__()

        # Three bi-LSTM branches with different hidden dimensions
        self.lstm1 = nn.LSTM(input_dim, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_dim, 128, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(input_dim, 64, bidirectional=True, batch_first=True)

        # Calculate total dimension after FLO concatenation
        # Bidirectional LSTM: hidden_dim * 2
        # FLO: (first + last) timesteps * 2
        total_dim = (256 * 2) * 2 + (128 * 2) * 2 + (64 * 2) * 2  # 1792

        # MLP for projection
        self.dense1 = nn.Linear(total_dim, 128)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(128, output_dim)
        self.task_mode = task_mode

        # Validate task_mode
        if task_mode not in ['contrastive', 'classification', 'regression']:
            raise ValueError(f"task_mode must be 'contrastive', 'classification', or 'regression', got '{task_mode}'")

    def downsample(self, x: torch.Tensor, factor: int = 2) -> torch.Tensor:
        """
        Downsample temporal dimension by given factor.

        Args:
            x: Input tensor (batch, time, features)
            factor: Downsampling factor

        Returns:
            Downsampled tensor (batch, time//factor, features)
        """
        return x[:, ::factor, :]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through projector.

        Args:
            x: Encoded features (batch, channels, time)

        Returns:
            Projected features (batch, output_dim)
        """
        # Permute for LSTM: (batch, time, features)
        x = x.permute(0, 2, 1)

        # Branch 1: Full resolution LSTM
        lstm1_out, _ = self.lstm1(x)
        # FLO: concatenate first and last timestep outputs
        flo1 = torch.cat([lstm1_out[:, 0, :], lstm1_out[:, -1, :]], dim=1)

        # Branch 2: 50% downsampled LSTM
        x_down2 = self.downsample(x, factor=2)
        lstm2_out, _ = self.lstm2(x_down2)
        flo2 = torch.cat([lstm2_out[:, 0, :], lstm2_out[:, -1, :]], dim=1)

        # Branch 3: 50% downsampled LSTM
        x_down3 = self.downsample(x, factor=2)
        lstm3_out, _ = self.lstm3(x_down3)
        flo3 = torch.cat([lstm3_out[:, 0, :], lstm3_out[:, -1, :]], dim=1)

        # Concatenate all FLO outputs
        x = torch.cat([flo1, flo2, flo3], dim=1)

        # MLP projection
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)

        # Apply task-specific output activation
        if self.task_mode == 'classification':
            x = F.log_softmax(x, dim=1)
        elif self.task_mode == 'regression':
            x = x.squeeze(-1)  # Remove last dimension for scalar output

        return x


class EEGContrastiveModel(nn.Module):
    """
    Complete model for contrastive learning: Encoder + Projector.

    This is the main model for pretraining. During contrastive learning,
    only the projector output is used for the loss. After pretraining,
    the encoder can be extracted and used for downstream tasks.

    Args:
        in_channels: Number of input channels (default: 129)
        repeat_n: Number of residual blocks in encoder (default: 4)
        n_samples: Number of time samples per epoch (default: 200)
        output_dim: Dimension of projection space (default: 128)
    """

    def __init__(
        self,
        in_channels: int = N_CHANS,
        repeat_n: int = 4,
        n_samples: int = SAMPLEPOINTS,
        output_dim: int = PROJECTOR_OUTPUT_DIM,
        enable_preprocessing: bool = True
    ):
        super(EEGContrastiveModel, self).__init__()
        self.enable_preprocessing = enable_preprocessing
        if enable_preprocessing:
            self.preprocessor = EEGPreprocessor(zscore_method='channel_wise')
        self.encoder = ConvolutionalEncoder(in_channels, repeat_n, n_samples)
        self.projector = Projector(input_dim=4, output_dim=output_dim)

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass through encoder and projector.

        Args:
            x: Input EEG (batch, channels, time)

        Returns:
            encoded: Encoder features (batch, 4, time_reduced)
            projected: Projected features (batch, output_dim)
        """
        if self.enable_preprocessing:
            x = self.preprocessor(x)
        encoded = self.encoder(x)
        projected = self.projector(encoded)
        return encoded, projected

    def get_encoder(self) -> nn.Module:
        """
        Extract encoder for downstream tasks.

        Returns:
            Encoder module (can be frozen for transfer learning)
        """
        return self.encoder
