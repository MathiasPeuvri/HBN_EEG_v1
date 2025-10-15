# ##########################################################################
# # CRL-based Submission for HBN EEG Challenge
# # ---------------------------
# The zip file needs to be single level depth!
# NO FOLDER
# my_submission.zip
# ├─ submission.py
# ├─ crl_encoder_best.pth
# ├─ regressor_challenge1_best.pth
# └─ regressor_challenge2_best.pth

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pathlib import Path

# ============================================================================
# CONFIGURATION - ADAPT THIS FOR EACH SUBMISSION
# ============================================================================
# Model checkpoint filenames (must be in the zip)
CRL_ENCODER_CHECKPOINT = "crl_encoder_best.pth"
CHALLENGE_1_CHECKPOINT = "regressor_response_time_crl_frozen_best.pth"
CHALLENGE_2_CHECKPOINT = "regressor_response_time_crl_frozen_best.pth"

# Architecture parameters (must match training config)
N_CHANS = 129                # Number of EEG channels
N_SAMPLES = 200              # Samples per window (2s at 100Hz)
PROJECTOR_OUTPUT_DIM = 128   # CRL projector dimension
ENABLE_PREPROCESSING = True  # Use EEGPreprocessor (average ref + zscore)
ENCODER_REPEAT_N = 4         # Number of residual blocks in encoder

# Regression head parameters
REGRESSOR_DROPOUT = 0.3      # Dropout for regression head

# ============================================================================
# PREPROCESSING
# ============================================================================
class EEGPreprocessor(nn.Module):
    """Minimal preprocessing: average reference + zscore normalization"""

    def __init__(self, zscore_method='channel_wise'):
        super().__init__()
        self.zscore_method = zscore_method

    def forward(self, x):
        """x: (batch, channels, time)"""
        device = x.device
        batch_size = x.shape[0]
        processed = []
        for i in range(batch_size):
            sample = x[i].cpu().numpy()
            sample = self._average_reference(sample)
            sample = self._zscore(sample)
            processed.append(torch.from_numpy(sample))
        return torch.stack(processed).to(device)

    @staticmethod
    def _average_reference(data):
        """Subtract average across channels"""
        return data - np.mean(data, axis=0, keepdims=True)

    def _zscore(self, data):
        """Z-score normalization"""
        if self.zscore_method == 'channel_wise':
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
        elif self.zscore_method == 'globalnorm':
            mean = np.mean(data)
            std = np.std(data)
        else:
            raise ValueError(f"Unknown zscore_method: {self.zscore_method}")

        std = np.where(std == 0, 1, std)
        return (data - mean) / std

# ============================================================================
# CRL ENCODER
# ============================================================================
class ConvolutionalEncoder(nn.Module):
    """Multi-branch CNN encoder with residual blocks"""

    def __init__(self, in_channels=N_CHANS, repeat_n=ENCODER_REPEAT_N, n_samples=N_SAMPLES):
        super().__init__()

        # Scale kernel sizes based on signal length
        scale_factor = n_samples / 4000.0
        self.k1 = max(4, int(128 * scale_factor))
        self.k2 = max(4, int(64 * scale_factor))
        self.k3 = max(2, int(16 * scale_factor))

        # Three parallel branches
        self.branch1 = nn.Conv1d(in_channels, 100, kernel_size=self.k1, padding=0)
        self.branch2 = nn.Conv1d(in_channels, 100, kernel_size=self.k2, padding=0)
        self.branch3 = nn.Conv1d(in_channels, 50, kernel_size=self.k3, padding=0)

        # Dense layer (100+100+50=250)
        self.dense1 = nn.Linear(250, 250)

        # Residual blocks
        self.repeat_blocks = nn.ModuleList([
            nn.Sequential(
                nn.ReLU(),
                nn.BatchNorm1d(250),
                nn.Conv1d(250, 250, kernel_size=64, padding=0)
            ) for _ in range(repeat_n)
        ])

        # Final layers
        self.final_relu = nn.ReLU()
        self.final_bn = nn.BatchNorm1d(250)
        self.final_conv = nn.Conv1d(250, 4, kernel_size=64, padding=0)

    def forward(self, x):
        """x: (batch, channels, time) -> (batch, 4, time_reduced)"""
        # Apply reflection padding for each branch
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

        # Concatenate branches
        x = torch.cat([x1, x2, x3], dim=1)

        # Dense layer
        x = x.permute(0, 2, 1)
        x = self.dense1(x)
        x = x.permute(0, 2, 1)

        # Residual blocks
        for block in self.repeat_blocks:
            residual = x
            x = F.pad(x, (32, 31), mode='reflect')
            x = block(x)
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

# ============================================================================
# CRL PROJECTOR
# ============================================================================
class Projector(nn.Module):
    """Multi-scale bi-LSTM projector with FLO strategy"""

    def __init__(self, input_dim=4, output_dim=PROJECTOR_OUTPUT_DIM, task_mode='contrastive'):
        super().__init__()

        # Three bi-LSTM branches
        self.lstm1 = nn.LSTM(input_dim, 256, bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(input_dim, 128, bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(input_dim, 64, bidirectional=True, batch_first=True)

        # MLP projection (total_dim = 1792)
        total_dim = (256 * 2) * 2 + (128 * 2) * 2 + (64 * 2) * 2
        self.dense1 = nn.Linear(total_dim, 128)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(128, output_dim)
        self.task_mode = task_mode

    def downsample(self, x, factor=2):
        """Downsample temporal dimension"""
        return x[:, ::factor, :]

    def forward(self, x):
        """x: (batch, channels, time) -> (batch, output_dim)"""
        x = x.permute(0, 2, 1)  # (batch, time, features)

        # Branch 1: Full resolution
        lstm1_out, _ = self.lstm1(x)
        flo1 = torch.cat([lstm1_out[:, 0, :], lstm1_out[:, -1, :]], dim=1)

        # Branch 2: 50% downsampled
        x_down2 = self.downsample(x, factor=2)
        lstm2_out, _ = self.lstm2(x_down2)
        flo2 = torch.cat([lstm2_out[:, 0, :], lstm2_out[:, -1, :]], dim=1)

        # Branch 3: 50% downsampled
        x_down3 = self.downsample(x, factor=2)
        lstm3_out, _ = self.lstm3(x_down3)
        flo3 = torch.cat([lstm3_out[:, 0, :], lstm3_out[:, -1, :]], dim=1)

        # Concatenate and project
        x = torch.cat([flo1, flo2, flo3], dim=1)
        x = self.dense1(x)
        x = self.relu(x)
        x = self.dense2(x)

        # Task-specific output
        if self.task_mode == 'classification':
            x = F.log_softmax(x, dim=1)
        elif self.task_mode == 'regression':
            x = x.squeeze(-1)

        return x


# ============================================================================
# FULL CRL MODEL
# ============================================================================
class EEGContrastiveModel(nn.Module):
    """Complete CRL model: Preprocessor + Encoder + Projector"""

    def __init__(self, in_channels=N_CHANS, repeat_n=ENCODER_REPEAT_N,
                 n_samples=N_SAMPLES, output_dim=PROJECTOR_OUTPUT_DIM,
                 enable_preprocessing=ENABLE_PREPROCESSING):
        super().__init__()
        self.enable_preprocessing = enable_preprocessing
        if enable_preprocessing:
            self.preprocessor = EEGPreprocessor(zscore_method='channel_wise')
        self.encoder = ConvolutionalEncoder(in_channels, repeat_n, n_samples)
        self.projector = Projector(input_dim=4, output_dim=output_dim)

    def forward(self, x):
        """x: (batch, channels, time) -> (encoded, projected)"""
        if self.enable_preprocessing:
            x = self.preprocessor(x)
        encoded = self.encoder(x)
        projected = self.projector(encoded)
        return encoded, projected

    def get_encoder(self):
        """Extract encoder for downstream tasks"""
        return self.encoder


# ============================================================================
# REGRESSION HEAD
# ============================================================================
class CRLRegressionHead(nn.Module):
    """Regression head for CRL encoder"""

    def __init__(self, encoder, freeze_encoder=True, dropout=REGRESSOR_DROPOUT):
        super().__init__()
        self.encoder = encoder

        if freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False

        # Use Projector in regression mode
        self.projector = Projector(
            input_dim=4,
            output_dim=1,
            task_mode='regression'
        )

    def forward(self, x):
        """x: (batch, channels, time) -> (batch, 1)"""
        features = self.encoder(x)  # (batch, 4, time_reduced)
        output = self.projector(features)  # (batch,)
        return output.unsqueeze(-1)  # (batch, 1) for compatibility


# ============================================================================
# SUBMISSION CLASS
# ============================================================================
class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        """Load model for Challenge 1 (response time prediction)"""
        # Load CRL encoder
        encoder_checkpoint = torch.load(
            str(Path(__file__).parent / CRL_ENCODER_CHECKPOINT),
            map_location=self.device,
            weights_only=False
        )

        # Create full CRL model
        full_model = EEGContrastiveModel(
            in_channels=N_CHANS,
            repeat_n=ENCODER_REPEAT_N,
            n_samples=N_SAMPLES,
            output_dim=PROJECTOR_OUTPUT_DIM,
            enable_preprocessing=ENABLE_PREPROCESSING
        )
        full_model.load_state_dict(encoder_checkpoint['model_state_dict'])
        encoder = full_model.get_encoder()

        # Create regression head
        model = CRLRegressionHead(
            encoder=encoder,
            freeze_encoder=True,
            dropout=REGRESSOR_DROPOUT
        ).to(self.device)

        # Load regression head weights
        reg_checkpoint = torch.load(
            str(Path(__file__).parent / CHALLENGE_1_CHECKPOINT),
            map_location=self.device,
            weights_only=False
        )
        model.load_state_dict(reg_checkpoint['model_state_dict'])

        return model.eval()

    def get_model_challenge_2(self):
        """Load model for Challenge 2 (externalizing score prediction)"""
        # Load CRL encoder
        encoder_checkpoint = torch.load(
            str(Path(__file__).parent / CRL_ENCODER_CHECKPOINT),
            map_location=self.device,
            weights_only=False
        )

        # Create full CRL model
        full_model = EEGContrastiveModel(
            in_channels=N_CHANS,
            repeat_n=ENCODER_REPEAT_N,
            n_samples=N_SAMPLES,
            output_dim=PROJECTOR_OUTPUT_DIM,
            enable_preprocessing=ENABLE_PREPROCESSING
        )
        full_model.load_state_dict(encoder_checkpoint['model_state_dict'])
        encoder = full_model.get_encoder()

        # Create regression head
        model = CRLRegressionHead(
            encoder=encoder,
            freeze_encoder=True,
            dropout=REGRESSOR_DROPOUT
        ).to(self.device)

        # Load regression head weights
        reg_checkpoint = torch.load(
            str(Path(__file__).parent / CHALLENGE_2_CHECKPOINT),
            map_location=self.device,
            weights_only=False
        )
        model.load_state_dict(reg_checkpoint['model_state_dict'])

        return model.eval()


# ##########################################################################
# # How Submission class will be used
# # ---------------------------------
# from submission import Submission
#
# SFREQ = 100
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# sub = Submission(SFREQ, DEVICE)
# model_1 = sub.get_model_challenge_1()
# model_1.eval()
#
# # Challenge 1: Response time prediction
# with torch.inference_mode():
#     for batch in warmup_loader_challenge_1:
#         X, y, infos = batch
#         X = X.to(dtype=torch.float32, device=DEVICE)
#         # X.shape is (BATCH_SIZE, 129, 200)
#         y_pred = model_1.forward(X)
#         # y_pred.shape is (BATCH_SIZE,)
#
# # Challenge 2: Externalizing score prediction
# model_2 = sub.get_model_challenge_2()
# model_2.eval()
#
# with torch.inference_mode():
#     for batch in warmup_loader_challenge_2:
#         X, y, crop_inds, infos = batch
#         X = X.to(dtype=torch.float32, device=DEVICE)
#         # X.shape is (BATCH_SIZE, 129, 200)
#         y_pred = model_2.forward(X)
#         # y_pred.shape is (BATCH_SIZE,)
