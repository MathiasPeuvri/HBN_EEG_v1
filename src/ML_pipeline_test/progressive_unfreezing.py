"""
Progressive Unfreezing Strategy for CRL Encoder Fine-tuning

Phase 1 (epochs 1-5): Encoder completely frozen, train only regression head
Phase 2 (epochs 6-20): Progressive unfreezing with differential learning rates
    - Epochs 6-8:   Unfreeze final_conv + repeat_blocks[3]
    - Epochs 9-11:  Unfreeze repeat_blocks[2-3]
    - Epochs 12-14: Unfreeze repeat_blocks[1-3]
    - Epochs 15+:   Unfreeze all encoder layers
"""

import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple


# ============================================================================
# CONFIGURATION - All progressive unfreezing parameters centralized here
# ============================================================================

FROZEN_EPOCHS = 5           # Number of initial epochs with frozen encoder
ENCODER_LR = 1e-5           # Learning rate for encoder (10-20x lower than head)
RECOMMENDED_TOTAL_EPOCHS = 20  # Recommended total epochs for full strategy


def get_unfreeze_schedule(frozen_epochs: int = FROZEN_EPOCHS) -> Dict[int, List[str]]:
    """
    Get the progressive unfreezing schedule.
        }
    """
    schedule = {
        frozen_epochs + 1: ['final_conv', 'final_bn', 'final_relu', 'repeat_blocks.3'],
        frozen_epochs + 4: ['repeat_blocks.2'],
        frozen_epochs + 7: ['repeat_blocks.1'],
        frozen_epochs + 10: ['repeat_blocks.0', 'dense1', 'branch1', 'branch2', 'branch3']
    }
    return schedule


def unfreeze_encoder_layers(model: nn.Module, layers_to_unfreeze: List[str], verbose: bool = True) -> int:
    """
    Unfreeze specific encoder layers by name pattern matching.

    """
    unfrozen_count = 0
    unfrozen_params = []

    # Access encoder from the model (model.encoder for both RegressionHead and CRLRegressionHead)
    if not hasattr(model, 'encoder'):
        raise ValueError("Model does not have an 'encoder' attribute")

    encoder = model.encoder
    # Iterate through all encoder parameters
    for name, param in encoder.named_parameters():
        # Check if this parameter matches any pattern in layers_to_unfreeze
        for pattern in layers_to_unfreeze:
            if pattern in name:
                if not param.requires_grad:  # Only unfreeze if currently frozen
                    param.requires_grad = True
                    unfrozen_count += param.numel()
                    unfrozen_params.append((name, param.numel()))
                    if verbose:
                        print(f"  ✓ Unfroze 'encoder.{name}' ({param.numel():,} params)")
                break

    if verbose:
        print(f"  → Total unfrozen: {unfrozen_count:,} parameters")

    return unfrozen_count


def setup_progressive_optimizer(
    model: nn.Module,
    base_lr: float,
    weight_decay: float,
    encoder_lr: float = ENCODER_LR
) -> optim.Optimizer:
    """
    Create optimizer with differential learning rates for encoder and head.
    """
    # Separate encoder and head parameters
    encoder_params = []
    head_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'encoder' in name:
                encoder_params.append(param)
            else:
                head_params.append(param)

    # Create parameter groups with differential learning rates
    param_groups = []

    if encoder_params:
        param_groups.append({
            'params': encoder_params,
            'lr': encoder_lr,
            'weight_decay': weight_decay
        })
        print(f"  → Encoder params: {sum(p.numel() for p in encoder_params):,} (lr={encoder_lr:.2e})")

    if head_params:
        param_groups.append({
            'params': head_params,
            'lr': base_lr,
            'weight_decay': weight_decay
        })
        print(f"  → Head params: {sum(p.numel() for p in head_params):,} (lr={base_lr:.2e})")

    # Create optimizer with parameter groups
    optimizer = optim.Adam(param_groups)

    return optimizer


def get_trainable_params_info(model: nn.Module) -> Tuple[int, int, List[str]]:
    """
    Get information about trainable parameters in the model.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    trainable_layer_names = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            trainable_layer_names.append(name)

    return total_params, trainable_params, trainable_layer_names


def print_unfreezing_plan(schedule: Dict[int, List[str]], frozen_epochs: int = 5):
    """
    Print a readable summary of the unfreezing plan.
    """
    print("\n" + "=" * 60)
    print("Progressive Unfreezing Strategy")
    print("=" * 60)
    print(f"Phase 1 (epochs 1-{frozen_epochs}): Encoder FROZEN, train head only")
    print(f"\nPhase 2 (epochs {frozen_epochs + 1}+): Progressive unfreezing")

    for epoch in sorted(schedule.keys()):
        layers = schedule[epoch]
        print(f"  Epoch {epoch:2d}: Unfreeze {', '.join(layers)}")

    print("=" * 60 + "\n")
