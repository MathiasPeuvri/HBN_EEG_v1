#!/bin/bash
# Auto-logging: save all output to log file
LOG_FILE="logs/crl_finetuning_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs
exec > >(tee -a "$LOG_FILE") 2>&1


echo "========================================="
echo "CRL finetuning Pipeline - Testing Mode"
echo "========================================="

# Activate virtual environment
source venv/bin/activate

################################################################################
# CONFIGURATION SECTION - Edit these variables for your needs
################################################################################

# Dataset configuration

# Training configuration (default: 1 epoch for quick testing)
DOWNSTREAM_EPOCHS="${DOWNSTREAM_EPOCHS:-20}"
DOWNSTREAM_BATCH="${DOWNSTREAM_BATCH:-256}"

# Shard configuration
CHALL2_SUBJECTS_PER_SHARD="${CHALL2_SUBJECTS_PER_SHARD:-10}"  # Challenge 2: 10 subjects per shard

# Directory paths
PROJECT_ROOT="/home/mts/HBN_EEG_v1"
DATA_DIR="${PROJECT_ROOT}/datasets"
MODEL_DIR="${PROJECT_ROOT}/src/ML_pipeline_test/saved_models"
DATABASE_DIR="${PROJECT_ROOT}/database"

# Enable/disable pipeline steps (comment out lines to skip steps)
RUN_CHALL1_STANDARD=false
RUN_CHALL1_WINDOWED=false
RUN_CHALL2_TRAINING=false
RUN_CHALL2_TRAINING_UNFROZEN=false
# RUN_EVALUATION=true              # Uncomment when ready to evaluate
RUN_PREPARE_SUBMISSION=true      # Uncomment when ready for submission
RUN_CHECK_SUBMISSION=true        # Uncomment to test submission locally

################################################################################
# PIPELINE EXECUTION
################################################################################

echo ""
echo "Configuration Summary:"
echo "  Release:              $RELEASE"
echo "  Downstream epochs:    $DOWNSTREAM_EPOCHS"
echo "  Pretraining batch:    $PRETRAINING_BATCH"
echo "  Downstream batch:     $DOWNSTREAM_BATCH"
echo "  Data directory:       $DATA_DIR"
echo "  Model directory:      $MODEL_DIR"
echo "========================================="

# Create necessary directories
mkdir -p "$DATA_DIR"
mkdir -p "$MODEL_DIR"


################################################################################
# PHASE 1.1: CHALL 1 Standard Approach
################################################################################
PIPELINE_START=$SECONDS
PHASE1a_START=$SECONDS

if [ "$RUN_CHALL1_STANDARD" = true ]; then
    echo ""
    echo "========================================="
    echo "PHASE 2.2: Challenge 1 - Standard Approach"
    echo "========================================="
    echo "Training response time regression, (Maxime datashards v1), CRL (frozen)"
    echo "  Epochs: $DOWNSTREAM_EPOCHS"
    echo "  Batch size: $DOWNSTREAM_BATCH"
    echo ""

    python -m src.ML_pipeline_test.regression \
        --encoder_type crl \
        --target response_time \
        --epochs "$DOWNSTREAM_EPOCHS" \
        --batch-size "$DOWNSTREAM_BATCH"

    if [ $? -eq 0 ]; then
        echo "✓ Challenge 1 standard training completed"
        echo "  Model: $MODEL_DIR/regressor_response_time_best.pth"
    else
        echo "⚠ Challenge 1 standard training failed (may need v1 data shards)"
    fi
fi

PHASE1a_DURATION=$((SECONDS - PHASE1a_START))
echo "Phase 1.1 duration: $(printf '%02d:%02d:%02d' $((PHASE1a_DURATION/3600)) $((PHASE1a_DURATION%3600/60)) $((PHASE1a_DURATION%60)))"
################################################################################
# PHASE 1.2: CHALL 1 Windowed RT Index Augmentation
################################################################################
PHASE1b_START=$SECONDS
if [ "$RUN_CHALL1_WINDOWED" = true ]; then
    echo ""
    echo "========================================="
    echo "PHASE 2.3: Challenge 1 - Windowed RT Index Augmentation"
    echo "========================================="
    echo "Training with in-window RT index localization, (Maxime datashards v2), CRL (frozen)"
    echo "  Epochs: $DOWNSTREAM_EPOCHS"
    echo "  Batch size: $DOWNSTREAM_BATCH"
    echo ""    

    python -m src.ML_pipeline_test.rt_samplepoint_regression \
        --encoder_type crl \
        --target rt_idx \
        --epochs "$DOWNSTREAM_EPOCHS" \
        --batch-size "$DOWNSTREAM_BATCH" 

    if [ $? -eq 0 ]; then
        echo "✓ Challenge 1 windowed training completed"
        echo "  Model: $MODEL_DIR/regressor_rt_idx_best.pth"
    else
        echo "⚠ Challenge 1 windowed training failed (may need v2 data shards)"
    fi
fi

PHASE1b_DURATION=$((SECONDS - PHASE1b_START))
echo "Phase 1.2 duration: $(printf '%02d:%02d:%02d' $((PHASE1b_DURATION/3600)) $((PHASE1b_DURATION%3600/60)) $((PHASE1b_DURATION%60)))"

################################################################################
# PHASE 2.1: CHALL 2 Externalizing Factor
################################################################################
PHASE2_START=$SECONDS
if [ "$RUN_CHALL2_TRAINING" = true ]; then
    echo ""
    echo "========================================="
    echo "PHASE 2.4: Challenge 2 - Externalizing Factor"
    echo "========================================="
    echo "Training externalizing factor regression..."
    echo "  Target: externalizing"
    echo "  Encoder: CRL"
    echo "  Epochs: $DOWNSTREAM_EPOCHS"
    echo "  Batch size: $DOWNSTREAM_BATCH"
    echo ""

    # Train with frozen encoder first
    echo "  Step 1: Training with frozen CRL encoder..."
    python -m src.ML_pipeline_test.regression \
        --encoder_type crl \
        --target externalizing \
        --epochs "$DOWNSTREAM_EPOCHS" \
        --batch-size "$DOWNSTREAM_BATCH" \
        # --data-pattern "${DATA_DIR}/chall2/challenge2_data_shard_*_${RELEASE}.pkl"

    if [ $? -eq 0 ]; then
        echo "✓ Challenge 2 frozen encoder training completed"
    else
        echo "⚠ Challenge 2 frozen encoder training failed"
    fi

    # Optional: Fine-tune with unfrozen encoder (usually better results)
    if [ "$RUN_CHALL2_TRAINING_UNFROZEN" = true ]; then
        echo ""
        echo "  Step 2: Fine-tuning with unfrozen CRL encoder..."
        python -m src.ML_pipeline_test.regression \
            --encoder_type crl \
            --target externalizing \
            --epochs "$DOWNSTREAM_EPOCHS" \
            --batch-size "$DOWNSTREAM_BATCH" \
            --unfreeze 
            # --data-pattern "${DATA_DIR}/chall2/challenge2_data_shard_*_${RELEASE}.pkl"

        if [ $? -eq 0 ]; then
            echo "✓ Challenge 2 fine-tuning completed"
            echo "  Model: $MODEL_DIR/regressor_externalizing_best.pth"
        else
            echo "⚠ Challenge 2 fine-tuning failed"
        fi
    fi
fi

PHASE2_DURATION=$((SECONDS - PHASE2_START))
echo "Phase 2.1 duration: $(printf '%02d:%02d:%02d' $((PHASE2_DURATION/3600)) $((PHASE2_DURATION%3600/60)) $((PHASE2_DURATION%60)))"

################################################################################
# PHASE 3: SUBMISSION ready?!
################################################################################
PHASE3_START=$SECONDS

if [ "$RUN_PREPARE_SUBMISSION" = true ]; then
    echo ""
    echo "========================================="
    echo "PHASE 3.2: Prepare Submission"
    echo "========================================="
    echo "Creating submission zip file..."
    echo ""

    python -m src.ML_pipeline_test.starterkit.generate_submission_zip \
        src/ML_pipeline_test/starterkit/submission.py \
        "$MODEL_DIR/crl_encoder_V1_best.pth" \
        "$MODEL_DIR/regressor_rt_idx_crl_best.pth" \
        "$MODEL_DIR/regressor_externalizing_crl_best.pth"
        # "$MODEL_DIR/regressor_response_time_crl_best.pth" \
    if [ $? -eq 0 ]; then
        echo "Submission zip created"
    else
        echo "Submission zip creation failed"
    fi
fi

if [ "$RUN_CHECK_SUBMISSION" = true ]; then
    echo ""
    echo "========================================="
    echo "PHASE 3.3: Test Submission Locally"
    echo "========================================="
    echo "Running local scoring on submission..."
    echo ""

    # Find the latest submission zip
    LATEST_SUBMISSION=$(ls -t submission_*.zip 2>/dev/null | head -1)

    if [ -z "$LATEST_SUBMISSION" ]; then
        echo "No submission zip found. Run PHASE 3.2 first."
        exit 1
    fi

    echo "Testing submission: $LATEST_SUBMISSION"

    python -m src.ML_pipeline_test.starterkit.startkit_localscoring \
        --submission-zip "$LATEST_SUBMISSION" \
        --data-dir "$DATABASE_DIR" \
        --output-dir ./test_submission_output #\
        #--fast-dev-run
        
    if [ $? -eq 0 ]; then
        echo "Local submission fast-dev-run test passed: $LATEST_SUBMISSION ready to be submitted"
    else
        echo "Local submission fast-dev-run test failed"
    fi
fi

PHASE3_DURATION=$((SECONDS - PHASE3_START))
echo "Phase 3.1 duration: $(printf '%02d:%02d:%02d' $((PHASE3_DURATION/3600)) $((PHASE3_DURATION%3600/60)) $((PHASE3_DURATION%60)))"

################################################################################
# PIPELINE COMPLETE
################################################################################
echo ""
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
PIPELINE_TOTAL_DURATION=$((SECONDS - PIPELINE_START))
echo "Pipeline total duration: $(printf '%02d:%02d:%02d' $((PIPELINE_TOTAL_DURATION/3600)) $((PIPELINE_TOTAL_DURATION%3600/60)) $((PIPELINE_TOTAL_DURATION%60)))"
echo ""
echo "Configuration used:"
echo "  Downstream epochs:    $DOWNSTREAM_EPOCHS"
echo " Output Trained models location:       $MODEL_DIR/"
echo ""
echo "========================================="