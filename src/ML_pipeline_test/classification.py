"""Downstream classification for button press events"""
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from . import config
from .data_loader_ml import create_dataloaders
from .models import CNN1DAutoencoder, BinaryClassifier

"""
  1. load_pretrained_encoder: Creates fresh autoencoder → loads pretrained
  weights → returns encoder part for transfer learning
  2. downstream_train_epoch/validate: Standard supervised training loop
  with accuracy metrics, nearly identical to pretraining pattern
  3. train_classifier: Main pipeline - loads pretrained encoder, creates
  classifier, trains with frozen/unfrozen encoder options
  4. Flexibility improvements: Added autoencoder_class and classifier_class
   parameters for architectural flexibility
"""

def load_pretrained_encoder(autoencoder_class=CNN1DAutoencoder):
    """Load pretrained encoder from autoencoder
    !! This is what defined the model class used lated"""
    try:
        checkpoint = torch.load(config.AUTOENCODER_PATH, map_location=config.DEVICE)
        autoencoder = autoencoder_class()
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained autoencoder from epoch {checkpoint['epoch']}")
        return autoencoder.encoder
    except FileNotFoundError:
        print("Warning: No pretrained model found. Using random initialization.")
        return None


def downstream_train_epoch(model, train_loader, optimizer, criterion, epoch, epochs_n = config.CLS_EPOCHS):
    """Train for one epoch
    
        Minor note:
    The function is nearly identical to the pretraining train_epoch (same training pattern (forward → loss → backward → metrics))
    Can consider a shared training utility function to reduce code duplication. 
    Still optional - separate functions makes code more explicit and easier to customize per task."""

    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs_n} [Train]')
    
    for batch_idx, (data, labels) in enumerate(progress_bar):
        data, labels = data.to(config.DEVICE), labels.to(config.DEVICE)
        
        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        preds = torch.argmax(outputs, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Update progress bar
        if batch_idx % config.LOG_INTERVAL == 0:
            avg_loss = total_loss / (batch_idx + 1)
            acc = accuracy_score(all_labels, all_preds)
            progress_bar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{acc:.3f}'})
    
    train_acc = accuracy_score(all_labels, all_preds)
    # For IterableDataset, we need to count batches manually
    return total_loss / max(batch_idx + 1, 1), train_acc


def validate(model, val_loader, criterion, verbose=config.VERBOSE):
    """Validate the model - same comment as downstream_train_epoch, close to the pretraining validate"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    batch_count = 0
    
    with torch.no_grad():
        for data, labels in tqdm(val_loader, desc='Validation'):
            data, labels = data.to(config.DEVICE), labels.to(config.DEVICE)
            
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, labels)
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            batch_count += 1
    
    val_acc = accuracy_score(all_labels, all_preds)
    
    # Print detailed metrics
    if verbose:
        print("\nValidation Metrics:")
        print(f"Accuracy: {val_acc:.3f}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, 
                                target_names=['Left Button', 'Right Button']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(all_labels, all_preds))
    
    # For IterableDataset, we need to count batches manually
    return total_loss / max(batch_count, 1), val_acc

def train_classifier(autoencoder_class=CNN1DAutoencoder, 
                      classifier_class=BinaryClassifier,
                      dataset_type='classification',
                      epochs=config.CLS_EPOCHS, batch_size=config.CLS_BATCH_SIZE, 
                     freeze_encoder=True):
    """Main training function for classifier"""
    print("=" * 50)
    print("Starting Classifier Training")
    print("=" * 50)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        dataset_type=dataset_type, batch_size=batch_size)
    
    # Load pretrained encoder
    encoder = load_pretrained_encoder(autoencoder_class=autoencoder_class)
    
    # Initialize model
    model = classifier_class(encoder=encoder, freeze_encoder=freeze_encoder).to(config.DEVICE)
    #print(f"Model initialized on {config.DEVICE}")
    
    # TODO: suppress this count and print to reduce code complexity // or verbosity
    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Encoder frozen: {freeze_encoder} --> Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer and criterion
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.CLS_LEARNING_RATE, weight_decay=config.CLS_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5)
    
    # Training loop
    best_val_acc = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(epochs):
        # Train
        train_loss, train_acc = downstream_train_epoch(model, train_loader, optimizer, criterion, epoch, epochs_n=epochs)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step(val_acc)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.6f}, "
              f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
            }, config.CLASSIFIER_PATH)
            print(f"Saved best model with val_acc: {val_acc:.3f}")
    
    print("\n" + "=" * 50)
    print("Training Complete!")
    print(f"Best validation accuracy: {best_val_acc:.3f}")
    print(f"Model saved to: {config.CLASSIFIER_PATH}")
    
    # Load and validate best model
    print("\nFinal validation with best model:")
    model.load_state_dict(torch.load(config.CLASSIFIER_PATH)['model_state_dict'])
    validate(model, val_loader, criterion, verbose=True)
    print("=" * 50)
    
    return model, train_losses, val_losses, train_accs, val_accs


def main():
    parser = argparse.ArgumentParser(description='Train classifier')
    parser.add_argument('--epochs', type=int, default=config.CLS_EPOCHS,
                       help='Number of epochs to train')
    parser.add_argument('--batch-size', type=int, default=config.CLS_BATCH_SIZE,
                       help='Batch size for training')
    parser.add_argument('--unfreeze', action='store_true',
                       help='Unfreeze encoder for fine-tuning')
    args = parser.parse_args()
    
    torch.manual_seed(config.RANDOM_SEED); np.random.seed(config.RANDOM_SEED) # Set seeds
    
    # Train model
    model, train_losses, val_losses, train_accs, val_accs = train_classifier(
        epochs=args.epochs, batch_size=args.batch_size, freeze_encoder=not args.unfreeze)
    
    # Print final statistics
    print(f"\nFinal Training Accuracy: {train_accs[-1]:.3f}")
    print(f"Final Validation Accuracy: {val_accs[-1]:.3f}")
    print(f"Best Validation Accuracy: {max(val_accs):.3f}")

if __name__ == "__main__":
    main()