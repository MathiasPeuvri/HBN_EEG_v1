#!/bin/bash
# Auto-logging: save all output to log file
LOG_FILE="logs/crl_training_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "$LOG_FILE") 2>&1

################################################################################
# Default: Release R1, 1 epoch per step for rapid testing
################################################################################
echo "========================================="
echo "CRL pretraining Pipeline - Testing Mode"
echo "========================================="

# Activate virtual environment
source venv/bin/activate
# Directory paths
PROJECT_ROOT="/home/mpeuvrier/HBN_EEG_v1"
    DATA_DIR="${PROJECT_ROOT}/datasets"
    MODEL_DIR="${PROJECT_ROOT}/src/ML_pipeline_test/saved_models"
    DATABASE_DIR="${PROJECT_ROOT}/database"
# Create necessary directories
mkdir -p "$DATA_DIR"
mkdir -p "$MODEL_DIR"
mkdir -p "$DATABASE_DIR"


PRETRAINING_BATCH="${PRETRAINING_BATCH:-1000}" # tenter 512 / 1024 ? juste quelques secondes voir si crash ? 

# objectif :  répartir 60 heure pour les deux models. 
# Estimé 1 h par epoch onerelease, 6 h par epoch multirelease.
ONERELEASE="R1"
PRETRAINING_ONERELEASE_EPOCHS="${PRETRAINING_ONERELEASE_EPOCHS:-50}" #""
PRETRAINING_ONERELEASE_EARLYSTOPPING="${PRETRAINING_ONERELEASE_EARLYSTOPPING:-10}"

MULTIRELEASE="R1 R2 R3"
PRETRAINING_MULTIRELEASE_EPOCHS="${PRETRAINING_MULTIRELEASE_EPOCHS:-15}" #"15" # on déborde largement sur les 60 heures... early stopping, prayers and hope...
PRETRAINING_MULTIRELEASE_EARLYSTOPPING="${PRETRAINING_MULTIRELEASE_EARLYSTOPPING:-5}"

# --warmup-epochs 5 & 3 ? -> 7 & 4 car augmentation lr 
# --lr (3e-4 actuel) ->  6e-4  voir 1e-3 ? -> 1e-3 boost


################################################################################
# PHASE 1: MODEL TRAINING ONE RELEASE
################################################################################
PIPELINE_START=$SECONDS
PHASE1_START=$SECONDS

DATA_PATTERN="${DATA_DIR}/crl_pretraining_data_shard_*_${ONERELEASE}.pkl"
echo ""
echo "========================================="
echo "PHASE 1: CRL Pretraining ONE RELEASE"
echo "========================================="
echo "Training CRL encoder : "
echo "  Epochs: $PRETRAINING_ONERELEASE_EPOCHS"
echo "  Batch size: $PRETRAINING_BATCH"
echo "  Output: $MODEL_DIR/crl_encoder_${ONERELEASE}_best.pth"
echo ""

python -m src.ML_pipeline_test.crl_pretraining \
    --epochs "$PRETRAINING_ONERELEASE_EPOCHS" \
    --batch-size "$PRETRAINING_BATCH" \
    --data-pattern "$DATA_PATTERN" \
    --save-prefix "crl_encoder_${ONERELEASE}"\
    --early-stopping "$PRETRAINING_ONERELEASE_EARLYSTOPPING" \
    --warmup-epochs 7 \
    --lr 6e-4 \
    --grad-clip 2.0

PHASE1_DURATION=$((SECONDS - PHASE1_START))

if [ $? -eq 0 ]; then
    echo "CRL pretraining completed"
    echo "  Best model: $MODEL_DIR/crl_encoder_${ONERELEASE}_best.pth"
    echo "  Last checkpoint: $MODEL_DIR/crl_encoder_${ONERELEASE}_last.pth"
    echo "  Phase 1.1 duration: $(printf '%02d:%02d:%02d' $((PHASE1_DURATION/3600)) $((PHASE1_DURATION%3600/60)) $((PHASE1_DURATION%60)))"
else
    echo "CRL pretraining failed"
    exit 1
fi



################################################################################
# PHASE 2: MODEL TRAINING MULTI RELEASE
################################################################################
PHASE2_START=$SECONDS

DATA_PATTERN="${DATA_DIR}/crl_pretraining_data_shard_*.pkl"
echo ""
echo "========================================="
echo "PHASE 2: CRL Pretraining MULTI RELEASE"
echo "========================================="
echo "Training CRL encoder : "
echo "  Epochs: $PRETRAINING_MULTIRELEASE_EPOCHS"
echo "  Batch size: $PRETRAINING_BATCH"
echo "  Output: $MODEL_DIR/crl_encoder_${MULTIRELEASE}_best.pth"
echo ""

python -m src.ML_pipeline_test.crl_pretraining \
    --epochs "$PRETRAINING_MULTIRELEASE_EPOCHS" \
    --batch-size "$PRETRAINING_BATCH" \
    --data-pattern "$DATA_PATTERN" \
    --save-prefix "crl_encoder_${MULTIRELEASE}"\
    --early-stopping "$PRETRAINING_MULTIRELEASE_EARLYSTOPPING"\
    --warmup-epochs 4 \
    --lr 6e-4 \
    --grad-clip 2.0

PHASE2_DURATION=$((SECONDS - PHASE2_START))

if [ $? -eq 0 ]; then
    echo "CRL pretraining completed"
    echo "  Best model: $MODEL_DIR/crl_encoder_${MULTIRELEASE}_best.pth"
    echo "  Last checkpoint: $MODEL_DIR/crl_encoder_${MULTIRELEASE}_last.pth"
    echo "  Phase 2.1 duration: $(printf '%02d:%02d:%02d' $((PHASE2_DURATION/3600)) $((PHASE2_DURATION%3600/60)) $((PHASE2_DURATION%60)))"
else
    echo "CRL pretraining failed"
    exit 1
fi


################################################################################
# PIPELINE COMPLETE
################################################################################
PIPELINE_TOTAL_DURATION=$((SECONDS - PIPELINE_START))

echo ""
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo ""
echo "Timing Summary:"
if [ -n "${PHASE1_DURATION+x}" ]; then
    echo "Phase 1: (CRL Pretraining ONE RELEASE):    $(printf '%02d:%02d:%02d' $((PHASE1_DURATION/3600)) $((PHASE1_DURATION%3600/60)) $((PHASE1_DURATION%60)))"
fi
if [ -n "${PHASE2_DURATION+x}" ]; then
    echo "PHASE 2: CRL Pretraining MULTI RELEASE:  $(printf '%02d:%02d:%02d' $((PHASE2_DURATION/3600)) $((PHASE2_DURATION%3600/60)) $((PHASE2_DURATION%60)))"
fi
echo "========================================="
echo "  Total Pipeline Duration:      $(printf '%02d:%02d:%02d' $((PIPELINE_TOTAL_DURATION/3600)) $((PIPELINE_TOTAL_DURATION%3600/60)) $((PIPELINE_TOTAL_DURATION%60)))"
echo ""
echo "========================================="