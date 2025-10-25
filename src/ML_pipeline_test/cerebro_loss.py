"""
Cerebro Loss for Masked Autoencoder (MAE) pretraining
Implements the loss function from the paper with separate components for masked and visible positions.
"""
import torch
import torch.nn as nn


class CerebroMAELoss(nn.Module):
    """
    MAE Loss from Cerebro article.

    Loss = L_masked + alpha * L_visible

    Where:
    - L_masked = (1/|M|) × Σ ||P - P̂||²  on masked positions
    - L_visible = (1/|M̄|) × Σ ||P - P̂||²  on visible positions
    - alpha: weighting coefficient for visible loss (default: 0.1)
    """

    def __init__(self, alpha=0.1):
        super(CerebroMAELoss, self).__init__()
        self.alpha = alpha

    def forward(self, reconstruction, target, mask):
        """
        Compute the MAE loss with separate masked and visible components.

        Args:
            reconstruction (torch.Tensor): Reconstructed signal from the model (batch_size, channels, sequence_length)
            target (torch.Tensor): Original signal (ground truth) (batch_size, channels, sequence_length)
            mask (torch.Tensor): Binary mask from masking_strategy.py True = VISIBLE False = MASKED

        Returns:
            torch.Tensor: Combined loss value
            dict: Dictionary containing individual loss components for logging
                  {'loss': total_loss, 'loss_masked': L_masked, 'loss_visible': L_visible}
        """
        # Handle different mask shapes from masking_strategy.py:
        # - 'timepoint'/'block': (batch, seq_len) → expand to (batch, channels, seq_len)
        # - 'channel': already (batch, channels, seq_len)
        if mask.dim() == 2:
            mask = mask.unsqueeze(1).expand_as(reconstruction)

        # Mask convention: True = visible, False = masked
        visible_mask = mask.bool()
        masked_mask = ~visible_mask
        # Compute squared errors: ||P - P̂||²
        squared_errors = (reconstruction - target) ** 2
        # L_masked = (1/|M|) × Σ ||P - P̂||² on masked positions
        masked_errors = squared_errors[masked_mask]
        loss_masked = masked_errors.mean() if masked_errors.numel() > 0 else torch.tensor(0.0, device=reconstruction.device)
        # L_visible = (1/|M̄|) × Σ ||P - P̂||² on visible positions
        visible_errors = squared_errors[visible_mask]
        loss_visible = visible_errors.mean() if visible_errors.numel() > 0 else torch.tensor(0.0, device=reconstruction.device)
        # Combined loss: L = L_masked + alpha * L_visible
        total_loss = loss_masked + self.alpha * loss_visible
        loss_components = {'loss': total_loss, 'loss_masked': loss_masked,'loss_visible': loss_visible}

        return total_loss, loss_components

    def __repr__(self):
        return f"CerebroMAELoss(alpha={self.alpha})"


# Example usage and testing
if __name__ == "__main__":
    from masking_strategy import create_and_apply_mask

    print("=" * 60)
    print("Testing CerebroMAELoss with different masking strategies")
    print("=" * 60)

    # Test parameters
    batch_size = 4
    channels = 129
    seq_length = 200
    mask_ratio = 0.25

    # Create dummy data
    target = torch.randn(batch_size, channels, seq_length)
    reconstruction = target + torch.randn_like(target) * 0.1  # Add some noise

    # Initialize loss
    criterion = CerebroMAELoss(alpha=0.1)
    print(f"\nLoss function: {criterion}\n")

    # Test with different masking strategies
    strategies = ['timepoint', 'channel', 'block']

    for strategy in strategies:
        print(f"\n{'─' * 60}")
        print(f"Strategy: {strategy.upper()}")
        print(f"{'─' * 60}")

        # Create mask using masking_strategy.py
        masked_data, mask = create_and_apply_mask(
            target,
            mask_ratio=mask_ratio,
            strategy=strategy,
            block_size=10
        )

        # Compute loss
        loss, components = criterion(reconstruction, target, mask)

        # Calculate masking statistics
        total_elements = mask.numel()
        visible_elements = mask.sum().item()
        masked_elements = total_elements - visible_elements
        visible_pct = 100 * visible_elements / total_elements
        masked_pct = 100 * masked_elements / total_elements

        print(f"Mask shape: {mask.shape}")
        print(f"Total elements: {total_elements}")
        print(f"Visible (True): {int(visible_elements)} ({visible_pct:.1f}%)")
        print(f"Masked (False): {int(masked_elements)} ({masked_pct:.1f}%)")
        print(f"\nLoss components:")
        print(f"  Total loss:   {loss.item():.6f}")
        print(f"  Masked loss:  {components['loss_masked'].item():.6f}")
        print(f"  Visible loss: {components['loss_visible'].item():.6f}")
        print(f"  Contribution: L_masked + {criterion.alpha} × L_visible")

    print(f"\n{'=' * 60}")
    print("Testing complete!")
    print("=" * 60)
