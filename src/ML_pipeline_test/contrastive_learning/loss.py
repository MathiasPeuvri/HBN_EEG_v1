"""
NT-Xent Loss for Contrastive Learning
Implements Normalized Temperature-scaled Cross Entropy loss used in SimCLR

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
        """
        Initialize NT-Xent loss.
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature if temperature is not None else TEMPERATURE

    def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss for a batch of positive pairs.
        """
        batch_size = z_i.shape[0]
        device = z_i.device

        # Step 1: Concatenate both views into single tensor
        z = torch.cat([z_i, z_j], dim=0)
        # Step 2: Normalize embeddings to unit length (for cosine similarity)
        z_norm = F.normalize(z, dim=1)
        # Step 3: Compute similarity matrix for all pairs
        sim_matrix = torch.mm(z_norm, z_norm.t()) / self.temperature
        # Step 4: Create mask for positive pairs
        # For each sample in z_i, its positive is the corresponding sample in z_j
        positive_mask = torch.zeros((2*batch_size, 2*batch_size), dtype=torch.bool, device=device)
        positive_mask[torch.arange(batch_size), torch.arange(batch_size) + batch_size] = True
        positive_mask[torch.arange(batch_size) + batch_size, torch.arange(batch_size)] = True
        # Step 5: Create mask for negative pairs
        # Negatives are all samples except self and positive pair
        diagonal_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=device)
        negatives_mask = ~diagonal_mask & ~positive_mask
        # Step 6: Extract similarities
        # Positives: (2*batch_size, 1) - one positive per sample
        positives = sim_matrix[positive_mask].view(2 * batch_size, 1)
        # Negatives: (2*batch_size, 2*batch_size - 2) - all other samples
        negatives = sim_matrix[negatives_mask].view(2 * batch_size, -1)
        # Step 7: Create logits for cross-entropy
        # Concatenate positive and negatives
        logits = torch.cat([positives, negatives], dim=1)
        # Step 8: Labels for cross-entropy
        # Positive is always at index 0 after concatenation
        labels = torch.zeros(2 * batch_size, dtype=torch.long, device=device)
        # Step 9: Compute cross-entropy loss
        # This maximizes similarity to positive, minimizes to negatives
        loss = F.cross_entropy(logits, labels)

        return loss


# class NTXentLossWithWeighting(NTXentLoss):
#     """
#     Extended NT-Xent loss with hard negative weighting.

#     Upweights hard negatives (samples with high similarity to anchor)
#     to improve representation learning.

#     Args:
#         temperature: Temperature parameter
#         beta: Hard negative weighting factor (default: 1.0, no weighting)
#     """

#     def __init__(self, temperature: float = None, beta: float = 1.0):
#         """
#         Initialize weighted NT-Xent loss.

#         Args:
#             temperature: Temperature parameter
#             beta: Hard negative weight (beta > 1 emphasizes hard negatives)
#         """
#         super().__init__(temperature)
#         self.beta = beta

#     def forward(self, z_i: torch.Tensor, z_j: torch.Tensor) -> torch.Tensor:
#         """
#         Compute weighted NT-Xent loss.

#         Args:
#             z_i: Projections of view 1
#             z_j: Projections of view 2

#         Returns:
#             Weighted NT-Xent loss
#         """
#         if self.beta == 1.0:
#             # No weighting, use standard NT-Xent
#             return super().forward(z_i, z_j)

#         # TODO: Implement hard negative weighting
#         # For now, fall back to standard NT-Xent
#         # This can be extended in the future if needed
#         return super().forward(z_i, z_j)
