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
PROJECT_ROOT="/home/mts/HBN_EEG_v1"
    DATA_DIR="${PROJECT_ROOT}/datasets"
    MODEL_DIR="${PROJECT_ROOT}/src/ML_pipeline_test/saved_models"
    DATABASE_DIR="${PROJECT_ROOT}/database"

# python -m src.database_to_dataset.database_to_crl_pretraining_shards_EEGChallengeDataset --release R4 --cache-dir "$DATABASE_DIR
# " --savepath-root "$DATA_DIR" --subjects-per-shard 5  --verbose

# Dataset configuration
RELEASES="${RELEASES:-R1}"                    # Space-separated list: "R1 R2 R5"
VAL_DATA_PATTERN="${VAL_DATA_PATTERN:-${DATA_DIR}/crl_pretraining_data_shard_*2_R2.pkl}"
# si releases est une liste, vérifie uniquement que les shards spécifiés existent; fait l'entrainement sur toutes les shards existante peut importe la release
IFS=' ' read -ra RELEASE_ARRAY <<< "$RELEASES"

# Training configuration - set defaults based on number of releases
if [ -z "$PRETRAINING_EPOCHS" ]; then
    if [ ${#RELEASE_ARRAY[@]} -eq 1 ]; then # Only set default if not already specified by user
        PRETRAINING_EPOCHS=1  # Single release: more epochs
    else
        PRETRAINING_EPOCHS=1  # Multiple releases: quick test
    fi
fi

PRETRAINING_BATCH="${PRETRAINING_BATCH:-256}"



# Shard configuration
SUBJECTS_PER_SHARD="${SUBJECTS_PER_SHARD:-5}"  # CRL pretraining: 5 subjects per shard (heavy with augmentations)

# Enable/disable pipeline steps (comment out lines to skip steps)
RUN_CREATE_CRL_SHARDS=true
RUN_CRL_PRETRAINING=False


################################################################################
# PIPELINE EXECUTION
################################################################################

# Initialize timing
PIPELINE_START=$SECONDS

echo ""
echo "CRL pretraining config summary:"
echo "  Releases:             $RELEASES"
echo "  Subjects per shard:   $SUBJECTS_PER_SHARD"
# echo "  Database directory:   $DATABASE_DIR"
# echo "  Datasets directory:   $DATA_DIR"
echo ""
echo "  Pretraining epochs:   $PRETRAINING_EPOCHS"
echo "  Pretraining batch:    $PRETRAINING_BATCH"
# echo "  Model directory:      $MODEL_DIR"
echo "========================================="

# Create necessary directories
mkdir -p "$DATA_DIR"
mkdir -p "$MODEL_DIR"
mkdir -p "$DATABASE_DIR"

# Check if data already exists
echo ""
echo "Checking for existing data..."
RELEASES_TO_CREATE=()

for RELEASE in "${RELEASE_ARRAY[@]}"; do
    CRL_SHARDS=$(ls ${DATA_DIR}/crl_pretraining_data_shard_*_${RELEASE}.pkl 2>/dev/null | wc -l)
    
    if [ $CRL_SHARDS -gt 0 ]; then
        echo "  $RELEASE: Found $CRL_SHARDS shard(s) → skip"
    else
        echo "  $RELEASE: No shards → will create"
        RELEASES_TO_CREATE+=("$RELEASE")
    fi
done

################################################################################
# PHASE 1: DATA PREPARATION
################################################################################

if [ ${#RELEASES_TO_CREATE[@]} -gt 0 ]; then
    echo ""
    echo "========================================="
    echo "PHASE 1.1: Creating CRL Pretraining Shards"
    echo "========================================="
    echo "Creating shards for releases: ${RELEASES_TO_CREATE[@]}"
    echo ""

    PHASE1_START=$SECONDS
    
    # Loop through each release that needs shards
    for RELEASE in "${RELEASES_TO_CREATE[@]}"; do
        echo ""
        echo "--- Processing release: $RELEASE ---"
        
        python -m src.database_to_dataset.database_to_crl_pretraining_shards_EEGChallengeDataset \
            --release "$RELEASE" \
            --cache-dir "$DATABASE_DIR" \
            --savepath-root "$DATA_DIR" \
            --subjects-per-shard "$SUBJECTS_PER_SHARD" \
            --verbose

        if [ $? -ne 0 ]; then
            echo "Failed to create shards for release $RELEASE"
            exit 1
        fi
    done

    PHASE1_DURATION=$((SECONDS - PHASE1_START))
    
    echo ""
    echo "All shards created successfully"
    echo "Phase 1.1 duration: $(printf '%02d:%02d:%02d' $((PHASE1_DURATION/3600)) $((PHASE1_DURATION%3600/60)) $((PHASE1_DURATION%60)))"
else
    echo ""
    echo "All required shards already exist → Skipping Phase 1.1"
fi

################################################################################
# PHASE 2: MODEL TRAINING
################################################################################

if [ "$RUN_CRL_PRETRAINING" = true ]; then
    echo ""
    echo "========================================="
    echo "PHASE 2.1: CRL Pretraining"
    echo "========================================="
    echo "Training CRL encoder : "
    echo "  Epochs: $PRETRAINING_EPOCHS"
    echo "  Batch size: $PRETRAINING_BATCH"
        # Determine data pattern based on number of releases
    if [ ${#RELEASE_ARRAY[@]} -eq 1 ]; then # Single release: use specific pattern
        DATA_PATTERN="${DATA_DIR}/crl_pretraining_data_shard_*_${RELEASE_ARRAY[0]}.pkl"
        echo "  Data pattern: $DATA_PATTERN  (single release)"
    else # Multiple releases: use general pattern to include all
        DATA_PATTERN="${DATA_DIR}/crl_pretraining_data_shard_*.pkl"
        echo "  Data pattern: $DATA_PATTERN  (multiple releases)"
    fi
    
    echo "  Output: $MODEL_DIR/crl_encoder_best.pth"
    echo ""

    PHASE2_START=$SECONDS

    # Disable tqdm progress bars for cleaner logs
    # export TQDM_DISABLE=1
    
    python -m src.ML_pipeline_test.crl_pretraining \
        --epochs "$PRETRAINING_EPOCHS" \
        --batch-size "$PRETRAINING_BATCH" \
        --data-pattern "$DATA_PATTERN" \
        --val-data-pattern "$VAL_DATA_PATTERN" \
        --save-prefix "crl_encoder_${RELEASES}"

    PHASE2_DURATION=$((SECONDS - PHASE2_START))
    
    if [ $? -eq 0 ]; then
        echo "CRL pretraining completed"
        echo "  Best model: $MODEL_DIR/crl_encoder_best.pth"
        echo "  Last checkpoint: $MODEL_DIR/crl_encoder_last.pth"
        echo "  Phase 2.1 duration: $(printf '%02d:%02d:%02d' $((PHASE2_DURATION/3600)) $((PHASE2_DURATION%3600/60)) $((PHASE2_DURATION%60)))"
    else
        echo "CRL pretraining failed"
        exit 1
    fi
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
echo "Releases:             $RELEASES"
echo "Pretraining epochs:   $PRETRAINING_EPOCHS"
echo "Output  Data shards:          $DATA_DIR/"
echo "Output  Trained models:       $MODEL_DIR/"
echo ""
echo "Timing Summary:"
if [ -n "${PHASE1_DURATION+x}" ]; then
    echo "  Phase 1.1 (Create Shards):    $(printf '%02d:%02d:%02d' $((PHASE1_DURATION/3600)) $((PHASE1_DURATION%3600/60)) $((PHASE1_DURATION%60)))"
fi
if [ -n "${PHASE2_DURATION+x}" ]; then
    echo "  Phase 2.1 (CRL Pretraining):  $(printf '%02d:%02d:%02d' $((PHASE2_DURATION/3600)) $((PHASE2_DURATION%3600/60)) $((PHASE2_DURATION%60)))"
fi
echo "========================================="
echo "  Total Pipeline Duration:      $(printf '%02d:%02d:%02d' $((PIPELINE_TOTAL_DURATION/3600)) $((PIPELINE_TOTAL_DURATION%3600/60)) $((PIPELINE_TOTAL_DURATION%60)))"
echo ""
echo ""
echo "========================================="