# ##########################################################################
# # Example of submission files
# # ---------------------------
# The zip file needs to be single level depth!
# NO FOLDER
# my_submission.zip
# ├─ submission.py
# ├─ weights_challenge_1.pt
# └─ weights_challenge_2.pt

import torch
import torch.nn as nn

import torch.nn.functional as F
from pathlib import Path



# Model paths
MODEL_DIR = Path(__file__).parent / "./saved_models"
AUTOENCODER_PATH = MODEL_DIR / "autoencoder_best.pth"
CLASSIFIER_PATH = MODEL_DIR / "classifier_best.pth"

# Data configuration
NUM_CHANNELS = 129

# Model architecture
CONV1_OUT_CHANNELS = 64
CONV2_OUT_CHANNELS = 32
CONV3_OUT_CHANNELS = 16
KERNEL_SIZE = 5
CLS_DROPOUT = 0.3

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% 
# Auto encodeur based on the masking SSL task
class RegressionHead(nn.Module):
    """Regression head for continuous value prediction"""
    def __init__(self, encoder=None, freeze_encoder=True, dropout=CLS_DROPOUT):
        super().__init__()
        self.encoder = encoder
        self.freeze_encoder = freeze_encoder
        
        if encoder is not None and freeze_encoder:
            for param in encoder.parameters():
                param.requires_grad = False
        
        # Feature extraction (same as BinaryClassifier)
        self.feature_extractor = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling
            nn.Flatten()
        )
        
        # Regression head (similar to BinaryClassifier but with 1 output)
        self.regressor = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(CONV3_OUT_CHANNELS, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)  # Single output for regression
        )
        
    def forward(self, x):
        if self.encoder is not None:
            features = self.encoder(x) # Extract features 
        else:
            features = x
        # Apply feature extraction (pooling + flattening) then Regress to single value
        features = self.feature_extractor(features)
        return self.regressor(features) #.squeeze(-1)  # Remove last dimension


class CNN1DAutoencoder(nn.Module):
    """3-layer 1D CNN Autoencoder with masking capability"""
    
    def __init__(self):
        super(CNN1DAutoencoder, self).__init__()
        
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(NUM_CHANNELS, CONV1_OUT_CHANNELS, 
                     kernel_size=KERNEL_SIZE, stride=1, padding=2),
            nn.ReLU(),
            nn.Conv1d(CONV1_OUT_CHANNELS, CONV2_OUT_CHANNELS,
                     kernel_size=KERNEL_SIZE, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(CONV2_OUT_CHANNELS, CONV3_OUT_CHANNELS,
                     kernel_size=KERNEL_SIZE, stride=2, padding=2),
            nn.ReLU()
        )
        
        # Decoder layers (mirror architecture)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(CONV3_OUT_CHANNELS, CONV2_OUT_CHANNELS,
                              kernel_size=KERNEL_SIZE, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(CONV2_OUT_CHANNELS, CONV1_OUT_CHANNELS,
                              kernel_size=KERNEL_SIZE, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(CONV1_OUT_CHANNELS, NUM_CHANNELS,
                              kernel_size=KERNEL_SIZE, stride=1, padding=2)
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


def load_pretrained_encoder(autoencoder_class=CNN1DAutoencoder):
    """Load pretrained encoder from autoencoder"""
    try:
        checkpoint = torch.load(AUTOENCODER_PATH, map_location=DEVICE)
        autoencoder = autoencoder_class()
        autoencoder.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded pretrained autoencoder from epoch {checkpoint['epoch']}")
        return autoencoder.encoder
    except FileNotFoundError:
        print("Warning: No pretrained model found. Using random initialization.")
        return None


# %% Submission class
class Submission:
    def __init__(self, SFREQ, DEVICE):
        self.sfreq = SFREQ
        self.device = DEVICE

    def get_model_challenge_1(self):
        encoder = load_pretrained_encoder(autoencoder_class=CNN1DAutoencoder)
        model = RegressionHead(encoder=encoder, freeze_encoder=True).to(DEVICE)
        checkpoint_1 = torch.load(Path(__file__).parent / "regressor_p_factor_best.pth", map_location=self.device)
        model.load_state_dict(checkpoint_1['model_state_dict'])

        return model


    def get_model_challenge_2(self):
        return self.get_model_challenge_1()



# ##########################################################################
# # How Submission class will be used
# # ---------------------------------
# from submission import Submission
#
# SFREQ = 100
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# sub = Submission(SFREQ, DEVICE)
# model_1 = sub.get_model_challenge_1()
# model_1.eval()

# warmup_loader_challenge_1 = DataLoader(HBN_R5_dataset1, batch_size=BATCH_SIZE)
# final_loader_challenge_1 = DataLoader(secret_dataset1, batch_size=BATCH_SIZE)

# with torch.inference_mode():
#     for batch in warmup_loader_challenge_1:  # and final_loader later
#         X, y, infos = batch
#         X = X.to(dtype=torch.float32, device=DEVICE)
#         # X.shape is (BATCH_SIZE, 129, 200)

#         # Forward pass
#         y_pred = model_1.forward(X)
#         # save prediction for computing evaluation score
#         ...
# score1 = compute_score_challenge_1(y_true, y_preds)
# del model_1
# gc.collect()

# model_2 = sub.get_model_challenge_2()
# model_2.eval()

# warmup_loader_challenge_2 = DataLoader(HBN_R5_dataset2, batch_size=BATCH_SIZE)
# final_loader_challenge_2 = DataLoader(secret_dataset2, batch_size=BATCH_SIZE)

# with torch.inference_mode():
#     for batch in warmup_loader_challenge_2:  # and final_loader later
#         X, y, crop_inds, infos = batch
#         X = X.to(dtype=torch.float32, device=DEVICE)
#         # X shape is (BATCH_SIZE, 129, 200)

#         # Forward pass
#         y_pred = model_2.forward(X)
#         # save prediction for computing evaluation score
#         ...
# score2 = compute_score_challenge_2(y_true, y_preds)
# overall_score = compute_leaderboard_score(score1, score2)