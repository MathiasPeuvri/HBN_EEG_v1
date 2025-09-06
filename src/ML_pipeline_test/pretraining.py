"""
Masked autoencoder pretraining for EEG data
"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from . import config
from .data_loader_ml import create_dataloaders
from .models import CNN1DAutoencoder

"""
● Pretraining.py Summary 

  - Main pipeline: train_autoencoder() is the complete training workflow - data
  loading, model initialization, training loop, and best model selection
  - Training loop: train_epoch() handles forward pass with masking, MSE loss
  computation, and backpropagation (zero_grad → backward → step)
  - Validation: Runs model on unseen data without parameter updates to detect
  overfitting and guide model selection/learning rate scheduling
  - Your architectural insights: TODOs suggest moving masking to data
  preprocessing and using standard forward() instead of forward_masked() for
  better separation of concerns
  - Backward pass mechanics: zero_grad() clears old gradients, backward() computes
   new gradients via backpropagation, step() updates parameters using optimizer
  - Loss function concern: Current MSE loss computes against all positions, but
  MAE should focus loss on masked positions only
  - Learning rate scheduler: ReduceLROnPlateau automatically reduces learning rate
   when validation loss stops improving
  - Model persistence: Saves best model based on validation loss, not training
  loss, ensuring better generalization
"""
# TODO : add early stopping ?
# TODO change train_epoch to MAE_training_epoch, check the loss function, add data masking instead of using raw data and use the model.forward instead of specific masked forward function

def train_epoch(model, train_loader, optimizer, epoch, epochs_n = config.AE_EPOCHS, verbose=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    criterion = nn.MSELoss() # TODO: check it is the right loss function / adapt it to the task
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs_n} [Train]')
    
    for batch_idx, data in enumerate(progress_bar):
        data = data.to(config.DEVICE)
        
        # Forward pass with masking
        reconstruction, mask = model.forward_masked(data, mask_ratio=config.MASK_RATIO)
        
        # Compute loss only on masked positions
        loss = criterion(reconstruction, data) # check the loss function as we want higher loss for reconstruction of masked data (but still small score for the unmasked reconstruction)
        
        # Backward pass
        optimizer.zero_grad() #reset the gradient calculator
        loss.backward() #Computes gradients of the loss
        optimizer.step() #updates the model parameters using the computed gradients / new_weight = old_weight - learning_rate * gradient
        
        # Track loss
        total_loss += loss.item()
        
        # Update progress bar
        if batch_idx % config.LOG_INTERVAL == 0:
            avg_loss = total_loss / (batch_idx + 1)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}'})
    if verbose:
        print(f"Input data range: {data.min():.6f} to {data.max():.6f}")
        print(f"Reconstruction range: {reconstruction.min():.6f} to {reconstruction.max():.6f}")
        print(f"Raw loss: {loss.item():.8f}")
    
    #return total_loss / len(val_loader) # if it is not an IterableDataset

    # For IterableDataset, we need to count batches manually
    return total_loss / max(batch_idx + 1, 1)


def validate(model, val_loader):
    """Runs the model on validation data without updating parameters
    Computes loss on unseen data to measure generalization performance """
    model.eval()
    total_loss = 0
    criterion = nn.MSELoss()
    batch_count = 0
    
    with torch.no_grad():
        for data in tqdm(val_loader, desc='Validation'):
            data = data.to(config.DEVICE)
            
            # Forward pass with masking
            reconstruction, mask = model.forward_masked(data, mask_ratio=config.MASK_RATIO)
            
            # Compute loss
            loss = criterion(reconstruction, data)
            total_loss += loss.item()
            batch_count += 1
    
    #return total_loss / len(val_loader) # if it is not an IterableDataset
    # For IterableDataset, we need to count batches manually
    return total_loss / max(batch_count, 1)


def train_autoencoder(model_class=CNN1DAutoencoder, dataset_type='pretraining', epochs=config.AE_EPOCHS, batch_size=config.AE_BATCH_SIZE, verbose=config.VERBOSE):
    """Main training function for autoencoder"""
    print("=" * 50)
    print("Starting Autoencoder Pretraining")
    print("=" * 50)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        dataset_type=dataset_type,
        batch_size=batch_size
    )
    
    # Initialize model
    model = model_class().to(config.DEVICE)
    print(f"Model initialized on {config.DEVICE}")
    
    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.AE_LEARNING_RATE,
        weight_decay=config.AE_WEIGHT_DECAY
    )
    
    # Learning rate scheduler : reduce learning rate when validation loss plateaus
    """ never used this approach, check if it is useful """
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, epoch, epochs_n=epochs, verbose=verbose)
        train_losses.append(train_loss) #gives history of train loss for each epoch
        
        # Validate
        val_loss = validate(model, val_loader)
        val_losses.append(val_loss) # gives history of val loss for each epoch
        
        # Update learning rate
        scheduler.step(val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
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
            #print(f"Saved best model with val_loss: {val_loss:.4f}")
            best_epoch = epoch
    
    print("\n" + "=" * 50)
    print("Selfsupervised Training Complete!")
    print(f"Best epoch: {best_epoch+1}, best validation loss: {best_val_loss:.4f}")
    print(f"Model saved to: {config.AUTOENCODER_PATH}")
    print("=" * 50)
    
    return model, train_losses, val_losses


def main():
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


if __name__ == "__main__":
    main()