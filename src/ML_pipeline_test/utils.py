"""
Utility functions for EEG ML pipeline
"""
import argparse
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from . import config
from .models import CNN1DAutoencoder, BinaryClassifier
from .data_loader_ml import DownstreamTaskDataset

""" post-training inspection tools 
This is prototype evaluation code!!!
  gets you started but would need substantial enhancement for real research or deployment. 
  The core training pipeline (pretraining.py, classification.py, models.py) is the valuable part 
  - utils.py is just temporary inspection tooling


"""

def evaluate_model(model_type='classifier'):
    """Evaluate trained model performance"""
    print("=" * 50)
    print(f"Evaluating {model_type.capitalize()}")
    print("=" * 50)
    
    if model_type == 'autoencoder':
        evaluate_autoencoder()
    elif model_type == 'classifier':
        evaluate_classifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def evaluate_autoencoder():
    """Evaluate autoencoder reconstruction quality"""
    # Load model
    checkpoint = torch.load(config.AUTOENCODER_PATH, map_location=config.DEVICE)
    model = CNN1DAutoencoder().to(config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']+1}")
    print(f"Training loss: {checkpoint['train_loss']:.4f}")
    print(f"Validation loss: {checkpoint['val_loss']:.4f}")
    
    # Load test data
    from data_loader_ml import SSL_Dataset
    dataset = SSL_Dataset()
    
    # Test reconstruction on few random samples
    n_samples = 5
    indices = np.random.choice(len(dataset), n_samples, replace=False)
    
    mse_scores = []
    for idx in indices:
        sample = dataset[idx].unsqueeze(0).to(config.DEVICE)
        
        with torch.no_grad():
            reconstruction = model(sample)
        
        mse = torch.nn.functional.mse_loss(reconstruction, sample).item()
        mse_scores.append(mse)
        print(f"Sample {idx}: MSE = {mse:.4f}")
    
    print(f"\nAverage MSE: {np.mean(mse_scores):.4f}")
    print("=" * 50)


def evaluate_classifier():
    """Evaluate classifier performance"""
    # Load model
    checkpoint = torch.load(config.CLASSIFIER_PATH, map_location=config.DEVICE)
    
    # Load pretrained encoder
    encoder_checkpoint = torch.load(config.AUTOENCODER_PATH, map_location=config.DEVICE)
    autoencoder = CNN1DAutoencoder()
    autoencoder.load_state_dict(encoder_checkpoint['model_state_dict'])
    
    # Create classifier with pretrained encoder
    model = BinaryClassifier(encoder=autoencoder.encoder, freeze_encoder=True).to(config.DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint['epoch']+1}")
    print(f"Training accuracy: {checkpoint['train_acc']:.3f}")
    print(f"Validation accuracy: {checkpoint['val_acc']:.3f}")
    
    # Load test data
    dataset = DownstreamTaskDataset()
    
    # Evaluate on full dataset
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for i in range(len(dataset)):
            signal, label = dataset[i]
            signal = signal.unsqueeze(0).to(config.DEVICE)
            
            output = model(signal)
            pred = torch.argmax(output, dim=1).cpu().numpy()[0]
            
            all_preds.append(pred)
            all_labels.append(label.numpy())
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='binary'
    )
    
    print(f"\nFull Dataset Performance:")
    print(f"  Accuracy: {accuracy:.3f}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall: {recall:.3f}")
    print(f"  F1 Score: {f1:.3f}")
    
    print("=" * 50)


def plot_training_history():
    """Plot training history if available"""
    # This would require saving history during training
    # Placeholder for visualization
    print("Training history plotting not yet implemented")


def main():
    parser = argparse.ArgumentParser(description='Utility functions')
    # parser.add_argument('--check-data', action='store_true',
    #                    help='Check and verify data files')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate trained models')
    parser.add_argument('--model-type', type=str, default='classifier',
                       choices=['autoencoder', 'classifier'],
                       help='Model type to evaluate')
    args = parser.parse_args()
    
    # if args.check_data:
    #     check_data()
    
    if args.evaluate:
        evaluate_model(args.model_type)


if __name__ == "__main__":
    main()