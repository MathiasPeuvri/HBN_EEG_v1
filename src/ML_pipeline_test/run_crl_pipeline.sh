#!/bin/bash
################################################################################
# CRL Pipeline - Simplified Testing Version
#
# Purpose: Train CRL encoder and downstream tasks with minimal configuration
# Default: Release R1, 1 epoch per step for rapid testing
################################################################################

echo "========================================="
echo "CRL EEG ML Pipeline - Testing Mode"
echo "========================================="

# Activate virtual environment
source venv_HBN_linux/bin/activate

################################################################################
# CONFIGURATION SECTION - Edit these variables for your needs
################################################################################

# Dataset configuration
RELEASE="${RELEASE:-R1}"                    # Dataset release (R1, R2, R5, etc.)
MINI_MODE="${MINI_MODE:-}"     # never asked for it (never use mini mode)            # Set to "--mini" for mini dataset testing

# Training configuration (default: 1 epoch for quick testing)
PRETRAINING_EPOCHS="${PRETRAINING_EPOCHS:-1}"
DOWNSTREAM_EPOCHS="${DOWNSTREAM_EPOCHS:-1}"
PRETRAINING_BATCH="${PRETRAINING_BATCH:-256}"
DOWNSTREAM_BATCH="${DOWNSTREAM_BATCH:-32}"

# Shard configuration
SUBJECTS_PER_SHARD="${SUBJECTS_PER_SHARD:-10}"  # CRL pretraining: 10 subjects per shard (heavy with augmentations)
CHALL2_SUBJECTS_PER_SHARD="${CHALL2_SUBJECTS_PER_SHARD:-10}"  # Challenge 2: 10 subjects per shard

# Directory paths
PROJECT_ROOT="/home/mts/HBN_EEG_v1"
    DATA_DIR="${PROJECT_ROOT}/datasets"
    MODEL_DIR="${PROJECT_ROOT}/src/ML_pipeline_test/saved_models"
    DATABASE_DIR="${PROJECT_ROOT}/database"

# Enable/disable pipeline steps (comment out lines to skip steps)
RUN_CREATE_CRL_SHARDS=true
RUN_CREATE_CHALL2_SHARDS=true
RUN_CRL_PRETRAINING=true
RUN_CHALL1_STANDARD=true
RUN_CHALL1_WINDOWED=true
RUN_CHALL2_TRAINING=true
# RUN_EVALUATION=true              # Uncomment when ready to evaluate
# RUN_PREPARE_SUBMISSION=true      # Uncomment when ready for submission
# RUN_CHECK_SUBMISSION=true        # Uncomment to test submission locally

################################################################################
# PIPELINE EXECUTION
################################################################################

echo ""
echo "Configuration Summary:"
echo "  Release:              $RELEASE"
echo "  Pretraining epochs:   $PRETRAINING_EPOCHS"
echo "  Downstream epochs:    $DOWNSTREAM_EPOCHS"
echo "  Pretraining batch:    $PRETRAINING_BATCH"
echo "  Downstream batch:     $DOWNSTREAM_BATCH"
echo "  Subjects per shard:   $SUBJECTS_PER_SHARD"
echo "  Mini mode:            ${MINI_MODE:-disabled}"
echo "  Data directory:       $DATA_DIR"
echo "  Model directory:      $MODEL_DIR"
echo "========================================="

# Create necessary directories
mkdir -p "$DATA_DIR"
mkdir -p "$MODEL_DIR"

################################################################################
# PHASE 1: DATA PREPARATION
################################################################################

if [ "$RUN_CREATE_CRL_SHARDS" = true ]; then
    echo ""
    echo "========================================="
    echo "PHASE 1.1: Creating CRL Pretraining Shards"
    echo "========================================="
    echo "Creating multi-task CRL pretraining shards from $RELEASE..."
    echo "  Output pattern: crl_pretraining_data_shard_*_${RELEASE}.pkl"
    echo ""

    python -m src.database_to_dataset.database_to_crl_pretraining_shards_EEGChallengeDataset \
        --release "$RELEASE" \
        --cache-dir "$DATABASE_DIR" \
        --savepath-root "$DATA_DIR" \
        --subjects-per-shard "$SUBJECTS_PER_SHARD" \
        --epoch-length 2.0 \
        --overlap 0.0 \
        $MINI_MODE \
        --verbose

    if [ $? -eq 0 ]; then
        echo "✓ CRL pretraining shards created successfully"
        echo "  Check: $DATA_DIR/crl_pretraining_data_shard_*_${RELEASE}.pkl"
    else
        echo "✗ Failed to create CRL pretraining shards"
        exit 1
    fi
fi

if [ "$RUN_CREATE_CHALL2_SHARDS" = true ]; then
    echo ""
    echo "========================================="
    echo "PHASE 1.2: Creating Challenge 2 Shards"
    echo "========================================="
    echo "Creating Challenge 2 (externalizing) shards from $RELEASE..."
    echo "  Task: contrastChangeDetection (4-second windows, 2-second stride)"
    echo "  Output pattern: challenge2_data_shard_*_${RELEASE}.pkl"
    echo ""

    python -m src.database_to_dataset.database_to_challenge2_shards_EEGChallengeDataset \
        --release "$RELEASE" \
        --cache-dir "$DATABASE_DIR" \
        --savepath-root "$DATA_DIR" \
        --subjects-per-shard "$CHALL2_SUBJECTS_PER_SHARD" \
        --window-length 4.0 \
        --window-stride 2.0 \
        $MINI_MODE \
        --verbose

    if [ $? -eq 0 ]; then
        echo "✓ Challenge 2 shards created successfully"
        echo "  Check: $DATA_DIR/challenge2_data_shard_*_${RELEASE}.pkl"
    else
        echo "✗ Failed to create Challenge 2 shards"
        exit 1
    fi
fi

################################################################################
# PHASE 2: MODEL TRAINING
################################################################################

if [ "$RUN_CRL_PRETRAINING" = true ]; then
    echo ""
    echo "========================================="
    echo "PHASE 2.1: CRL Pretraining"
    echo "========================================="
    echo "Training CRL encoder on multi-task data..."
    echo "  Epochs: $PRETRAINING_EPOCHS"
    echo "  Batch size: $PRETRAINING_BATCH"
    echo "  Data pattern: $DATA_DIR/crl_pretraining_data_shard_*_${RELEASE}.pkl"
    echo "  Output: $MODEL_DIR/crl_encoder_best.pth"
    echo ""

    python -m src.ML_pipeline_test.crl_pretraining \
        --epochs "$PRETRAINING_EPOCHS" \
        --batch-size "$PRETRAINING_BATCH" \
        --data-pattern "${DATA_DIR}/crl_pretraining_data_shard_*_${RELEASE}.pkl" \
        --save-prefix "crl_encoder_${RELEASE}"

    if [ $? -eq 0 ]; then
        echo "✓ CRL pretraining completed"
        echo "  Best model: $MODEL_DIR/crl_encoder_best.pth"
        echo "  Last checkpoint: $MODEL_DIR/crl_encoder_last.pth"
    else
        echo "✗ CRL pretraining failed"
        exit 1
    fi
fi

if [ "$RUN_CHALL1_STANDARD" = true ]; then
    echo ""
    echo "========================================="
    echo "PHASE 2.2: Challenge 1 - Standard Approach"
    echo "========================================="
    echo "Training response time regression (standard approach)..."
    echo "  Approach: Maxime datashards v1 (standard format)"
    echo "  Target: response_time"
    echo "  Encoder: CRL (frozen)"
    echo "  Epochs: $DOWNSTREAM_EPOCHS"
    echo "  Batch size: $DOWNSTREAM_BATCH"
    echo ""

    # Note: This requires Maxime's eval shards (not created in this pipeline yet)
    # You may need to prepare these separately or uncomment shard creation above

    python -m src.ML_pipeline_test.regression \
        --encoder_type crl \
        --target response_time \
        --epochs "$DOWNSTREAM_EPOCHS" \
        --batch-size "$DOWNSTREAM_BATCH" \
        --data-format v1

    if [ $? -eq 0 ]; then
        echo "✓ Challenge 1 standard training completed"
        echo "  Model: $MODEL_DIR/regressor_response_time_best.pth"
    else
        echo "⚠ Challenge 1 standard training failed (may need v1 data shards)"
    fi
fi

if [ "$RUN_CHALL1_WINDOWED" = true ]; then
    echo ""
    echo "========================================="
    echo "PHASE 2.3: Challenge 1 - Windowed RT Index Augmentation"
    echo "========================================="
    echo "Training with in-window RT index localization..."
    echo "  Approach: Maxime datashards v2 (windowed augmentation)"
    echo "  Target: rt_idx (sample point index within window)"
    echo "  Encoder: CRL (frozen)"
    echo "  Epochs: $DOWNSTREAM_EPOCHS"
    echo "  Batch size: $DOWNSTREAM_BATCH"
    echo ""

    # Note: This requires Maxime's v2 shards with window augmentation

    python -m src.ML_pipeline_test.regression \
        --encoder_type crl \
        --target rt_idx \
        --epochs "$DOWNSTREAM_EPOCHS" \
        --batch-size "$DOWNSTREAM_BATCH" \
        --data-format v2_windowed

    if [ $? -eq 0 ]; then
        echo "✓ Challenge 1 windowed training completed"
        echo "  Model: $MODEL_DIR/regressor_rt_idx_best.pth"
    else
        echo "⚠ Challenge 1 windowed training failed (may need v2 data shards)"
    fi
fi

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
        --data-pattern "${DATA_DIR}/challenge2_data_shard_*_${RELEASE}.pkl"

    if [ $? -eq 0 ]; then
        echo "✓ Challenge 2 frozen encoder training completed"
    else
        echo "⚠ Challenge 2 frozen encoder training failed"
    fi

    # Optional: Fine-tune with unfrozen encoder (usually better results)
    echo ""
    echo "  Step 2: Fine-tuning with unfrozen CRL encoder..."
    python -m src.ML_pipeline_test.regression \
        --encoder_type crl \
        --target externalizing \
        --unfreeze \
        --epochs "$DOWNSTREAM_EPOCHS" \
        --batch-size "$DOWNSTREAM_BATCH" \
        --data-pattern "${DATA_DIR}/challenge2_data_shard_*_${RELEASE}.pkl"

    if [ $? -eq 0 ]; then
        echo "✓ Challenge 2 fine-tuning completed"
        echo "  Model: $MODEL_DIR/regressor_externalizing_best.pth"
    else
        echo "⚠ Challenge 2 fine-tuning failed"
    fi
fi

################################################################################
# PHASE 3: EVALUATION AND SUBMISSION (commented by default)
################################################################################

if [ "$RUN_EVALUATION" = true ]; then
    echo ""
    echo "========================================="
    echo "PHASE 3.1: Model Evaluation"
    echo "========================================="
    echo "Evaluating trained models on independent release..."
    echo ""

    # First, create evaluation datasets if needed
    python -m src.ML_pipeline_test.evaluation.evaluation --create-datasets

    # Then run evaluation
    python -m src.ML_pipeline_test.evaluation.evaluation \
        --output-dir "$MODEL_DIR"

    if [ $? -eq 0 ]; then
        echo "✓ Evaluation completed"
        echo "  Check results in: $MODEL_DIR/evaluation_results/"
    else
        echo "⚠ Evaluation failed"
    fi
fi

if [ "$RUN_PREPARE_SUBMISSION" = true ]; then
    echo ""
    echo "========================================="
    echo "PHASE 3.2: Prepare Submission"
    echo "========================================="
    echo "Creating submission zip file..."
    echo ""

    python -m src.ML_pipeline_test.starterkit.generate_submission_zip \
        src/ML_pipeline_test/starterkit/submission.py \
        "$MODEL_DIR/crl_encoder_best.pth" \
        "$MODEL_DIR/regressor_response_time_crl_best.pth" \
        "$MODEL_DIR/regressor_externalizing_crl_best.pth"
    # python -m src.ML_pipeline_test.starterkit.generate_submission_zip \
    #     src/ML_pipeline_test/starterkit/submission.py \
    #     "$MODEL_DIR/crl_encoder_best.pth" \
    #     "$MODEL_DIR/regressor_response_time_best.pth" \
    #     "$MODEL_DIR/regressor_externalizing_best.pth"

    if [ $? -eq 0 ]; then
        echo "✓ Submission zip created"
        echo "  Check: submission_*.zip"
    else
        echo "⚠ Submission creation failed"
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
        echo "✗ No submission zip found. Run PHASE 3.2 first."
        exit 1
    fi

    echo "Testing submission: $LATEST_SUBMISSION"

    python -m src.ML_pipeline_test.starterkit.startkit_localscoring \
        --submission-zip "$LATEST_SUBMISSION" \
        --data-dir "$DATABASE_DIR" \
        --output-dir ./test_submission_output \
        --fast-dev-run

    if [ $? -eq 0 ]; then
        echo "✓ Local submission test passed"
        echo "  Results in: ./test_submission_output/"
    else
        echo "⚠ Local submission test failed"
    fi
fi

################################################################################
# PIPELINE COMPLETE
################################################################################

echo ""
echo "========================================="
echo "Pipeline Complete!"
echo "========================================="
echo ""
echo "Configuration used:"
echo "  Release:              $RELEASE"
echo "  Pretraining epochs:   $PRETRAINING_EPOCHS"
echo "  Downstream epochs:    $DOWNSTREAM_EPOCHS"
echo ""
echo "Output locations:"
echo "  Data shards:          $DATA_DIR/"
echo "  Trained models:       $MODEL_DIR/"
echo ""
echo "To run with different settings, use environment variables:"
echo "  RELEASE=R5 PRETRAINING_EPOCHS=200 DOWNSTREAM_EPOCHS=50 ./run_crl_pipeline.sh"
echo ""
echo "To enable evaluation and submission phases, uncomment lines in CONFIGURATION SECTION"
echo "========================================="