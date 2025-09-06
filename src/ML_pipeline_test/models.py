"""
Neural network models for EEG ML pipeline
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.ML_pipeline_test import config


class CNN1DAutoencoder(nn.Module):
    """3-layer 1D CNN Autoencoder with masking capability"""
    
    def __init__(self):
        super(CNN1DAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(config.NUM_CHANNELS, config.CONV1_OUT_CHANNELS, 
                     kernel_size=config.KERNEL_SIZE, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(config.CONV1_OUT_CHANNELS, config.CONV2_OUT_CHANNELS,
                     kernel_size=config.KERNEL_SIZE, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(config.CONV2_OUT_CHANNELS, config.CONV3_OUT_CHANNELS,
                     kernel_size=config.KERNEL_SIZE, stride=2, padding=2),
            nn.ReLU()
        )
        
        # Decoder layers (mirror architecture)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(config.CONV3_OUT_CHANNELS, config.CONV2_OUT_CHANNELS,
                              kernel_size=config.KERNEL_SIZE, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(config.CONV2_OUT_CHANNELS, config.CONV1_OUT_CHANNELS,
                              kernel_size=config.KERNEL_SIZE, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(config.CONV1_OUT_CHANNELS, config.NUM_CHANNELS,
                              kernel_size=config.KERNEL_SIZE, stride=1, padding=2)
        )
    
    def encode(self, x):
        """Encode input to latent representation"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to reconstruction"""
        return self.decoder(z)
    
    def forward(self, x):
        """Standard forward pass without masking"""
        z = self.encode(x)
        reconstruction = self.decode(z)
        # Ensure output matches input size
        if reconstruction.shape[-1] != x.shape[-1]:
            reconstruction = F.interpolate(reconstruction, size=x.shape[-1], mode='linear')
        return reconstruction
    
    def forward_masked(self, x, mask_ratio=config.MASK_RATIO):
        """Forward pass with random masking
        
                            If we move the create mask to another file, then we could discard the 
                            forward_masked function and only use forward but instead of x being the original 
                            data it would be the masked data.
        """
        batch_size, channels, seq_len = x.shape
        
        # Create random mask for time segments
        mask = self.create_mask(batch_size, seq_len, mask_ratio)
        
        # Apply mask (set masked positions to 0)
        x_masked = x.clone()
        for b in range(batch_size):
            x_masked[b, :, ~mask[b]] = 0
        
        # Encode and decode
        z = self.encode(x_masked)
        reconstruction = self.decode(z)
        
        # Ensure output matches input size
        if reconstruction.shape[-1] != x.shape[-1]:
            reconstruction = F.interpolate(reconstruction, size=x.shape[-1], mode='linear')
        
        return reconstruction, mask
    
    def create_mask(self, batch_size, seq_len, mask_ratio=0.75):
        """Create random binary mask for time segments // Should be moved to a different file later (and reworked, not just random!)"""
        num_masked = int(seq_len * mask_ratio)
        mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=config.DEVICE)
        
        for i in range(batch_size):
            # Random indices to mask
            masked_indices = torch.randperm(seq_len)[:num_masked]
            mask[i, masked_indices] = False
        
        return mask


class BinaryClassifier(nn.Module):
    """Binary classifier using pretrained encoder"""
    
    def __init__(self, encoder=None, freeze_encoder=True):
        super(BinaryClassifier, self).__init__()
        
        # Use provided encoder or create new one
        if encoder is not None:
            self.encoder = encoder
        else:
            autoencoder = CNN1DAutoencoder()
            self.encoder = autoencoder.encoder
        
        # Freeze encoder weights if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # Move encoder to device before calculating output size
        self.encoder = self.encoder.to(config.DEVICE)
        
        # Calculate encoder output size
        with torch.no_grad():
            dummy_input = torch.zeros(1, config.NUM_CHANNELS, config.POSTTRAINING_SEQ_LEN).to(config.DEVICE)
            encoder_output = self.encoder(dummy_input)
            encoder_output_size = encoder_output.shape[1] * encoder_output.shape[2]
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten(),
            nn.Dropout(config.CLS_DROPOUT),
            nn.Linear(config.CONV3_OUT_CHANNELS, 32),
            nn.ReLU(),
            nn.Dropout(config.CLS_DROPOUT),
            nn.Linear(32, 2)  # Binary classification (2 classes)
        )
    
    def forward(self, x):
        # Extract features with encoder
        features = self.encoder(x)
        
        # Classify
        output = self.classifier(features)
        return output
    
    def unfreeze_encoder(self):
        """Unfreeze encoder for fine-tuning"""
        for param in self.encoder.parameters():
            param.requires_grad = True


if __name__ == "__main__":
    # Test autoencoder
    print("Testing CNN1DAutoencoder...")
    autoencoder = CNN1DAutoencoder().to(config.DEVICE)
    
    # Test with pretraining data shape
    x = torch.randn(4, config.NUM_CHANNELS, config.PRETRAINING_SEQ_LEN).to(config.DEVICE)
    reconstruction = autoencoder(x)
    print(f"Input shape: {x.shape}, Reconstruction shape: {reconstruction.shape}")
    
    # Test masked forward
    reconstruction_masked, mask = autoencoder.forward_masked(x, mask_ratio=0.5)
    print(f"Masked reconstruction shape: {reconstruction_masked.shape}")
    print(f"Mask shape: {mask.shape}, Masked ratio: {(~mask).float().mean():.2f}")
    
    # Test classifier
    print("\nTesting BinaryClassifier...")
    classifier = BinaryClassifier(encoder=autoencoder.encoder).to(config.DEVICE)
    
    # Test with posttraining data shape
    x = torch.randn(4, config.NUM_CHANNELS, config.POSTTRAINING_SEQ_LEN).to(config.DEVICE)
    output = classifier(x)
    print(f"Classifier input shape: {x.shape}, Output shape: {output.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in autoencoder.parameters())
    trainable_params = sum(p.numel() for p in autoencoder.parameters() if p.requires_grad)
    print(f"\nAutoencoder parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    classifier_params = sum(p.numel() for p in classifier.classifier.parameters())
    print(f"Classifier head parameters: {classifier_params:,}")