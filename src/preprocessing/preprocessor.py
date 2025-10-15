"""EEG preprocessing module for integration into models."""

import torch
import torch.nn as nn
import numpy as np


class EEGPreprocessor(nn.Module):
    """Minimal preprocessing: average reference + zscore normalization."""

    def __init__(self, zscore_method: str = 'channel_wise'):
        super().__init__()
        self.zscore_method = zscore_method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, time)
        Returns:
            Preprocessed tensor (batch, channels, time)
        """
        device = x.device
        batch_size = x.shape[0]
        processed = []

        for i in range(batch_size):
            sample = x[i].cpu().numpy()  # (channels, time)
            sample = self._average_reference(sample)
            sample = self._zscore(sample)
            processed.append(torch.from_numpy(sample))

        return torch.stack(processed).to(device)

    @staticmethod
    def _average_reference(data: np.ndarray) -> np.ndarray:
        """Subtract average across channels."""
        return data - np.mean(data, axis=0, keepdims=True)

    def _zscore(self, data: np.ndarray) -> np.ndarray:
        """Z-score normalization."""
        if self.zscore_method == 'channel_wise':
            mean = np.mean(data, axis=1, keepdims=True)
            std = np.std(data, axis=1, keepdims=True)
        elif self.zscore_method == 'globalnorm':
            mean = np.mean(data)
            std = np.std(data)
        elif self.zscore_method == 'spatial':
            mean = np.mean(data, axis=0, keepdims=True)
            std = np.std(data, axis=0, keepdims=True)
        else:
            raise ValueError(f"Unknown zscore_method: {self.zscore_method}")

        std = np.where(std == 0, 1, std)
        return (data - mean) / std
