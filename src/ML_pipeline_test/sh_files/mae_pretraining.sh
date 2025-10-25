#!/bin/bash

LOG_FILE="logs/mae_pretraining_$(date +%Y%m%d_%H%M%S).log"
mkdir -p logs

START_TIME=$(date +%s)
python -m src.ML_pipeline_test.pretraining --epochs 50 --batch-size 256 2>&1 | tee >(grep -v $'\r' > "$LOG_FILE")
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

echo "Training completed in ${ELAPSED}s" | tee -a "$LOG_FILE"