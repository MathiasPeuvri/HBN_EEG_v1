"""
Configuration file for PyTorch EEG ML Pipeline
"""
import os
import glob
from pathlib import Path
import torch

# verbose
VERBOSE = False

# Project paths
# Détection auto racine du projet (ici src/ML_pipeline_test/config.py -> on remonte de 2 niveaux pour la racine)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
ML_DIR = PROJECT_ROOT / "src" / "ML_pipeline_test"
DATA_DIR = PROJECT_ROOT / "datasets"



#-------------------------------- Should rework the pathing stuff for model training -----------------------------
# Data paths
PRETRAINING_DATA_PATTERN = str(DATA_DIR / "pretraining_data_shard_*.pkl")
# Fallback to single file if no shards found
if not glob.glob(PRETRAINING_DATA_PATTERN):
    # Try batch pattern for backward compatibility
    PRETRAINING_DATA_PATTERN = str(DATA_DIR / "pretraining_data_batch_*.pkl")
    if not glob.glob(PRETRAINING_DATA_PATTERN):
        # Fallback to single file if no patterns match
        legacy_file = DATA_DIR / "pretraining_data.pkl"
        if legacy_file.exists():
            PRETRAINING_DATA_PATTERN = str(legacy_file)
        else:
            # Use original single batch file as last resort
            legacy_batch = DATA_DIR / "pretraining_data_batch_0.pkl"
            if legacy_batch.exists():
                PRETRAINING_DATA_PATTERN = str(legacy_batch)

# For backward compatibility, keep old name as alias
PRETRAINING_DATA_PATH = PRETRAINING_DATA_PATTERN

# Downstream data paths - support sharding
DOWNSTREAM_DATA_PATTERN = str(DATA_DIR / "challenge1_data_shard_*.pkl") # This is used for challenge 1 response time
DOWNSTREAM_DATA_PATTERN = str(DATA_DIR / "eval/R*.pkl")
#DOWNSTREAM_DATA_PATTERN = str(DATA_DIR / "posttraining_data_shard_*.pkl") # I think this is used for dataset created according to any behavioral task
#DOWNSTREAM_DATA_PATTERN = str(DATA_DIR / "pretraining_data_shard_*.pkl") # I think this is what we actually want for challenge 2
# Fallback to single file if no shards found
if not glob.glob(DOWNSTREAM_DATA_PATTERN):
    # Try generic downstream pattern
    DOWNSTREAM_DATA_PATTERN = str(DATA_DIR / "downstream_data_shard_*.pkl")
    if not glob.glob(DOWNSTREAM_DATA_PATTERN):
        DOWNSTREAM_DATA_PATTERN = str(DATA_DIR / "challenge_1_data*.pkl")
    if not glob.glob(DOWNSTREAM_DATA_PATTERN):
        # Fallback to single file if no patterns match
        legacy_file = DATA_DIR / "challenge_1_data.pkl"
        if legacy_file.exists():
            DOWNSTREAM_DATA_PATTERN = str(legacy_file)
        else:
            # Try posttraining_data.pkl as fallback
            legacy_posttraining = DATA_DIR / "posttraining_data.pkl"
            if legacy_posttraining.exists():
                DOWNSTREAM_DATA_PATTERN = str(legacy_posttraining)

# For backward compatibility, keep old name as alias
DOWNSTREAM_DATA_PATH = DOWNSTREAM_DATA_PATTERN
# Psychopathology data path
PARTICIPANTS_TSV_PATH = PROJECT_ROOT / "database/R1_L100/participants.tsv"
PSYCHOPATHOLOGY_FACTORS = ["p_factor", "attention", "internalizing", "externalizing"]

# Model paths
MODEL_DIR = ML_DIR / "saved_models"
AUTOENCODER_PATH = MODEL_DIR / "autoencoder_best.pth"
CLASSIFIER_PATH = MODEL_DIR / "classifier_best.pth"

# CRL model paths
CRL_ENCODER_PATH_BEST = MODEL_DIR / "crl_encoder_best.pth"
CRL_ENCODER_PATH_LAST = MODEL_DIR / "crl_encoder_last.pth"

# Data configuration
NUM_CHANNELS = 129
PRETRAINING_SEQ_LEN = 200
POSTTRAINING_SEQ_LEN = 200

# Training configuration
TRAIN_SPLIT = 0.8
RANDOM_SEED = 42

# Autoencoder configuration
AE_BATCH_SIZE = 32
AE_EPOCHS = 100
AE_LEARNING_RATE = 1e-3
AE_WEIGHT_DECAY = 1e-5
MASK_RATIO = 0.5

# Model architecture
CONV1_OUT_CHANNELS = 64
CONV2_OUT_CHANNELS = 32
CONV3_OUT_CHANNELS = 16
KERNEL_SIZE = 5

# Classifier configuration
CLS_BATCH_SIZE = 16
CLS_EPOCHS = 50
CLS_LEARNING_RATE = 1e-3
CLS_WEIGHT_DECAY = 1e-5
CLS_DROPOUT = 0.3

# Task configuration (set these to switch between tasks)
TASK_TYPE = 'classification'  # 'classification' or 'regression'
# TARGET_COLUMN = 'hit_accuracy'  # Column to extract targets from
# TARGET_EVENTS = ['correct', 'incorrect']  # Values to keep from TARGET_COLUMN (None = keep all)
TARGET_COLUMN = 'value'  # Column to extract targets from
TARGET_EVENTS = ['right_buttonPress', 'left_buttonPress']  # Values to keep from TARGET_COLUMN (None = keep all)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Logging
LOG_INTERVAL = 10
SAVE_INTERVAL = 10

# Create directories if they don't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# =========================== CRL Configuration ===========================
# For detailed CRL configuration, see:
# src/ML_pipeline_test/contrastive_learning/config.py
#
# Quick reference:
# - CRL uses multi-task pretraining (6 HBN tasks)
# - Data: 129 channels, 100Hz, 2s windows (200 samples)
# - Training: batch=256, epochs=200, temperature=0.1
# - To use CRL encoder in downstream tasks:
#     python regression.py --encoder_type crl --target response_time
