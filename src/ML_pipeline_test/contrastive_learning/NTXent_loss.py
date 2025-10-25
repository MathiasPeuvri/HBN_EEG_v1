"""
NT-Xent Loss for Contrastive Learning

References:
    - Chen et al. "A Simple Framework for Contrastive Learning of Visual
        Representations" (SimCLR, 2020)
    - Mohsenvand et al. "Contrastive Representation Learning for
        Electroencephalogram Classification" (2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .config import TEMPERATURE


class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss.
    Contrastive loss that pulls positive pairs together push negative pairs apart. Used for self-supervised learning.

    Mathematical formulation:
        For a positive pair (i, j):
        ℓ(i,j) = -log[ exp(sim(z_i, z_j)/τ) / Σ(k≠i) exp(sim(z_i, z_k)/τ) ]
        where:
        - sim(u,v) = cosine_similarity(u,v) = (u·v) / (||u|| × ||v||)
        - τ is the temperature parameter
    """

    def __init__(self, temperature: float = None):
        """ Initialize NT-Xent loss. """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature if temperature is not None else TEMPERATURE

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss for a batch of positive pairs.
        """
        batch_size = z_i.shape[0]
        device = z_i.device

        z = torch.cat([z_i, z_j], dim=0)
        z_norm = F.normalize(z, dim=1)
        sim_matrix = torch.mm(z_norm, z_norm.t()) / self.temperature
        # For each sample in z_i, its positive is the corresponding sample in z_j
        positive_mask = torch.zeros((2*batch_size, 2*batch_size), dtype=torch.bool, device=device)
        positive_mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = True
        positive_mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = True
        # Negatives are all samples except self and positive pair
        diagonal_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        negatives_mask = ~diagonal_mask & ~positive_mask
        # Positives: (2*batch_size, 1) - one positive per sample
        positives = sim_matrix[positive_mask].view(2 * batch_size, 1)
        # Negatives: (2*batch_size, 2*batch_size - 2) - all other samples
        negatives = sim_matrix[negatives_mask].view(2 * batch_size, -1)
        # Concatenate positive and negatives
        logits = torch.cat([positives, negatives], dim=1)
        # Positive is always at index 0 after concatenation
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=device)
        # This maximizes similarity to positive, minimizes to negatives
        loss = F.cross_entropy(logits, labels)

        return loss

