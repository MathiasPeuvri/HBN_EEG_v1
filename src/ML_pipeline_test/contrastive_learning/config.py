"""
Configuration for Contrastive Representation Learning (CRL)

Adapted from Mohsenvand et al. (2020) for HBN-EEG dataset
Original paper used: 200Hz, 4000 samples (20s), variable channels
HBN adaptation: 100Hz, 200 samples (2s), 129 channels
"""

# ================================ Data Configuration ================================
N_CHANS = 129  # 128 EEG channels + 1 reference
FS = 100  # Sampling frequency in Hz
SAMPLEPOINTS = 200  # Number of samples per epoch (2 seconds at 100Hz)

# ================================ Training Configuration ================================
BATCH_SIZE = 256  # Original paper uses 1000, reduced for hardware constraints
TEMPERATURE = 0.1  # Temperature parameter for NT-Xent loss (scaled with batch size)
# Note: Original paper uses τ=0.05 with batch=1000. τ should increase with smaller batches

LEARNING_RATE = 3e-4  # Initial learning rate
WEIGHT_DECAY = 1e-6  # L2 regularization
EPOCHS = 200  # Number of pretraining epochs
WARMUP_EPOCHS = 10  # Epochs for learning rate warmup

# Cosine annealing scheduler
MIN_LR = 1e-6  # Minimum learning rate for cosine annealing

# Gradient clipping
GRAD_CLIP = 1.0  # Maximum gradient norm

# ================================ Augmentation Ranges ================================
# Ranges adapted from original paper to shorter 2-second windows
transformation_ranges = {
    # Amplitude scaling: multiply signal by random factor
    'amplitude_scale': (0.5, 2.0),

    # Time shift: shift signal by ±10% of window (±0.2s = ±20 samples)
    'time_shift_percent_twindow': (-0.10, 0.10),

    # DC shift: add constant offset to signal (in µV)
    'DC_shift_µV': (-1.0, 1.0),

    # Zero masking: randomly set portion of signal to zero
    # Original: up to 0.75s, adapted to 5% of window (0.1s = 10 samples)
    'zero_masking_percent_twindow': (0.0, 0.05),

    # Additive Gaussian noise: scale relative to signal std
    'additive_gaussian_noise': (0.0, 0.2),

    # Band-stop filter: 5Hz bandwidth, random center frequency
    # Adapted for 100Hz: safe range 5-45Hz (avoid low/high frequency artifacts)
    'band5hz_stop_Hzstart': (5, 45),
}

# ================================ Model Architecture ================================
# Convolutional Encoder (auto-scales with SAMPLEPOINTS)
ENCODER_CHANNELS = [64, 128, 256]  # Progressive channel expansion
KERNEL_SIZE = 7  # Base kernel size (auto-scaled with n_samples)
STRIDE = 2  # Stride for downsampling
PADDING = 3  # Padding to maintain dimensions

# Projector (bi-LSTM)
PROJECTOR_HIDDEN_DIM = 256  # LSTM hidden dimension
PROJECTOR_NUM_LAYERS = 2  # Number of LSTM layers
PROJECTOR_OUTPUT_DIM = 128  # Final projection dimension for contrastive loss

# ================================ Data Loading ================================
NUM_WORKERS = 4  # Number of workers for data loading
PIN_MEMORY = True  # Pin memory for faster GPU transfer

# ================================ Logging & Checkpointing ================================
LOG_INTERVAL = 10  # Log every N batches
SAVE_INTERVAL = 10  # Save checkpoint every N epochs
CHECKPOINT_DIR = "saved_models"  # Directory for saving checkpoints

# ================================ CRL Configuration Dictionary ================================
CRL_CONFIG = {
    # Data
    'n_chans': N_CHANS,
    'fs': FS,
    'samplepoints': SAMPLEPOINTS,

    # Training
    'batch_size': BATCH_SIZE,
    'temperature': TEMPERATURE,
    'learning_rate': LEARNING_RATE,
    'weight_decay': WEIGHT_DECAY,
    'epochs': EPOCHS,
    'warmup_epochs': WARMUP_EPOCHS,
    'min_lr': MIN_LR,
    'grad_clip': GRAD_CLIP,

    # Augmentations
    'transformation_ranges': transformation_ranges,

    # Model
    'encoder_channels': ENCODER_CHANNELS,
    'kernel_size': KERNEL_SIZE,
    'stride': STRIDE,
    'padding': PADDING,
    'projector_hidden_dim': PROJECTOR_HIDDEN_DIM,
    'projector_num_layers': PROJECTOR_NUM_LAYERS,
    'projector_output_dim': PROJECTOR_OUTPUT_DIM,

    # Data loading
    'num_workers': NUM_WORKERS,
    'pin_memory': PIN_MEMORY,

    # Logging
    'log_interval': LOG_INTERVAL,
    'save_interval': SAVE_INTERVAL,
    'checkpoint_dir': CHECKPOINT_DIR,
}
