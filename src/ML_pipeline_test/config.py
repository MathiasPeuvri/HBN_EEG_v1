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
DOWNSTREAM_DATA_PATTERN = str(DATA_DIR / "challenge2_data_shard_*.pkl") # This is used for challenge 1 response time
DOWNSTREAM_CHALL1_PATTERN = str(DATA_DIR / "chall1/R*.pkl")
DOWNSTREAM_CHALL1_CLICKCENTERED_PATTERN = str(DATA_DIR / "chall1_clickcentered/R*_clickcentered.pkl")
#DOWNSTREAM_DATA_PATTERN = str(DATA_DIR / "posttraining_data_shard_*.pkl") # I think this is used for dataset created according to any behavioral task
#DOWNSTREAM_DATA_PATTERN = str(DATA_DIR / "pretraining_data_shard_*.pkl") # I think this is what we actually want for challenge 2
# Fallback to single file if no shards found
# if not glob.glob(DOWNSTREAM_DATA_PATTERN):
#     # Try generic downstream pattern
#     DOWNSTREAM_DATA_PATTERN = str(DATA_DIR / "downstream_data_shard_*.pkl")
#     if not glob.glob(DOWNSTREAM_DATA_PATTERN):
#         DOWNSTREAM_DATA_PATTERN = str(DATA_DIR / "challenge_1_data*.pkl")
#     if not glob.glob(DOWNSTREAM_DATA_PATTERN):
#         # Fallback to single file if no patterns match
#         legacy_file = DATA_DIR / "challenge_1_data.pkl"
#         if legacy_file.exists():
#             DOWNSTREAM_DATA_PATTERN = str(legacy_file)
#         else:
#             # Try posttraining_data.pkl as fallback
#             legacy_posttraining = DATA_DIR / "posttraining_data.pkl"
#             if legacy_posttraining.exists():
#                 DOWNSTREAM_DATA_PATTERN = str(legacy_posttraining)

# For backward compatibility, keep old name as alias
DOWNSTREAM_DATA_PATH = DOWNSTREAM_DATA_PATTERN

# Psychopathology data path - mapping from Release to OpenNeuro dataset
PSYCHOPATHOLOGY_FACTORS = ["p_factor", "attention", "internalizing", "externalizing"]

# Release to OpenNeuro dataset mapping (from eegdash)
RELEASE_TO_OPENNEURO_DATASET_MAP = {
    "R11": "ds005516",
    "R10": "ds005515",
    "R9": "ds005514",
    "R8": "ds005512",
    "R7": "ds005511",
    "R6": "ds005510",
    "R4": "ds005508",
    "R5": "ds005509",
    "R3": "ds005507",
    "R2": "ds005506",
    "R1": "ds005505",
}

def get_participants_tsv_for_release(release: str) -> Path:
    """
    Get the appropriate participants.tsv file for a given release.

    Args:
        release: Release identifier (e.g., "R1", "R2", etc.)

    Returns:
        Path to participants.tsv file

    Raises:
        FileNotFoundError: If participants.tsv not found for the release
    """
    if release not in RELEASE_TO_OPENNEURO_DATASET_MAP:
        raise ValueError(f"Unknown release: {release}. Expected one of {list(RELEASE_TO_OPENNEURO_DATASET_MAP.keys())}")

    openneuro_id = RELEASE_TO_OPENNEURO_DATASET_MAP[release]
    database_dir = PROJECT_ROOT / "database"

    # Try with -bdf suffix first (local download naming)
    participants_path = database_dir / f"{openneuro_id}-bdf" / "participants.tsv"
    if participants_path.exists():
        return participants_path

    # Try without -bdf suffix
    participants_path = database_dir / openneuro_id / "participants.tsv"
    if participants_path.exists():
        return participants_path

    raise FileNotFoundError(
        f"participants.tsv not found for {release} ({openneuro_id}) in {database_dir}"
    )

def get_all_participants_tsv_paths():
    """Get all available participants.tsv files in database directory."""
    database_dir = PROJECT_ROOT / "database"
    if not database_dir.exists():
        return []
    return sorted(database_dir.glob("*/participants.tsv"))

# Get all available participants.tsv paths
PARTICIPANTS_TSV_PATHS = get_all_participants_tsv_paths()

# For backward compatibility, use first available or None
PARTICIPANTS_TSV_PATH = PARTICIPANTS_TSV_PATHS[0] if PARTICIPANTS_TSV_PATHS else None

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
POSTTRAINING_SEQ_LEN = 200  # Standard window size (2s @ 100Hz)
CHALLENGE2_SEQ_LEN = 400     # Challenge 2 window size (4s @ 100Hz)

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
