"""
Masked autoencoder pretraining for EEG data
"""
import argparse
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import sys
sys.path.append('/home/mts/HBN_EEG_v1/src')
from . import config
from .data_loader_ml import create_dataloaders
from .models import CNN1DAutoencoder
from .cerebro_loss import CerebroMAELoss
from .masking_strategy import create_and_apply_mask
from preprocessing.preprocessor import EEGPreprocessor


# Masking parameters
MASKING_STRATEGY = 'block'  # Options: 'timepoint', 'channel', 'block'
BLOCK_SIZE = 10  # Only used for 'block' strategy
LOSS_ALPHA = 0.1  # Weight for visible loss component in CerebroMAELoss

def train_epoch(model, train_loader, optimizer, epoch, preprocessor=None, epochs_n = config.AE_EPOCHS, verbose=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    criterion = CerebroMAELoss(alpha=LOSS_ALPHA)

    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs_n} [Train]')

    for batch_idx, data in enumerate(progress_bar):
        data = data.to(config.DEVICE)
        # Apply preprocessing before masking
        if preprocessor is not None:
            with torch.no_grad():
                data = preprocessor(data)
        # Apply masking to preprocessed data
        masked_data, mask = create_and_apply_mask(data,
            mask_ratio=config.MASK_RATIO, strategy=MASKING_STRATEGY, block_size=BLOCK_SIZE)
        # Forward pass with masked data
        reconstruction = model(masked_data)
        loss, loss_components = criterion(reconstruction, data, mask)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Track loss
        total_loss += loss.item()
        if batch_idx % config.LOG_INTERVAL == 0:
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'}) # Update progress bar

    if verbose:
        print(f"Input data range: {data.min():.6f} to {data.max():.6f}")
        print(f"Reconstruction range: {reconstruction.min():.6f} to {reconstruction.max():.6f}")
        print(f"Loss breakdown - Total: {loss.item():.6f}, Masked: {loss_components['loss_masked'].item():.6f}, Visible: {loss_components['loss_visible'].item():.6f}")

    # For IterableDataset, we need to count batches manually
    return total_loss / max(batch_idx + 1, 1)


def validate(model, val_loader, preprocessor=None):
    """Runs the model on validation data without updating parameters
    Computes loss on unseen data to measure generalization performance """
    model.eval()

    total_loss = 0
    criterion = CerebroMAELoss(alpha=LOSS_ALPHA)
    batch_count = 0

    with torch.no_grad():
        for data in tqdm(val_loader, desc='Validation'):
            data = data.to(config.DEVICE)
            # Apply preprocessing before masking
            if preprocessor is not None:
                data = preprocessor(data)
            masked_data, mask = create_and_apply_mask(data,
                mask_ratio=config.MASK_RATIO, strategy=MASKING_STRATEGY, block_size=BLOCK_SIZE)
            # Forward pass with masked data
            reconstruction = model(masked_data)
            loss, loss_components = criterion(reconstruction, data, mask)
            total_loss += loss.item()
            batch_count += 1
    return total_loss / max(batch_count, 1)


def train_autoencoder(model_class=CNN1DAutoencoder, dataset_type='pretraining', epochs=config.AE_EPOCHS, batch_size=config.AE_BATCH_SIZE, use_preprocessing=True, verbose=config.VERBOSE):
    """Main training function for autoencoder"""
    print("=" * 50)
    print("Starting Autoencoder Pretraining")
    print("=" * 50)

    train_loader, val_loader = create_dataloaders(
        dataset_type=dataset_type, batch_size=batch_size)

    model = model_class().to(config.DEVICE)
    print(f"Model initialized on {config.DEVICE}")

    preprocessor = EEGPreprocessor(zscore_method='channel_wise').to(config.DEVICE) if use_preprocessing else None
    if preprocessor:
        print(f"Preprocessing enabled: average reference + channel-wise z-score")

    optimizer = optim.Adam(
        model.parameters(), lr=config.AE_LEARNING_RATE, weight_decay=config.AE_WEIGHT_DECAY)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, epoch, preprocessor=preprocessor, epochs_n=epochs, verbose=verbose)
        train_losses.append(train_loss) #gives history of train loss for each epoch
        # Validate
        val_loss = validate(model, val_loader, preprocessor=preprocessor)
        val_losses.append(val_loss) # gives history of val loss for each epoch
        # Update learning rate
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Current LR: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, config.AUTOENCODER_PATH)
            print(f"Saved best model (epoch {epoch+1}) with val_loss: {val_loss:.4f}")
            best_epoch = epoch
    
    print("\n" + "=" * 50)
    print(f"Best epoch: {best_epoch+1}, best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {config.AUTOENCODER_PATH}")
    print("=" * 50)
    
    return model, train_losses, val_losses


def main():
    import time
    start_time = time.time()
    parser = argparse.ArgumentParser(description='Train masked autoencoder')
    parser.add_argument('--epochs', type=int, default=config.AE_EPOCHS,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=config.AE_BATCH_SIZE,
                       help='Batch size for training')
    args = parser.parse_args()
    
    # Set random seeds for reproducibility
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)
    
    # Train model
    model, train_losses, val_losses = train_autoencoder(
        epochs=args.epochs,
        batch_size=args.batch_size
    )
    
    # Print final statistics
    print(f"\nFinal Training Loss: {train_losses[-1]:.4f}")
    print(f"Final Validation Loss: {val_losses[-1]:.4f}")
    print(f"Minimum Validation Loss: {min(val_losses):.4f}")
    print(f"Training time: {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()