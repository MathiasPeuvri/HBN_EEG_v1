#!/bin/bash

echo "========================================="
echo "PyTorch EEG ML Pipeline - Complete Test"
echo "========================================="

source venv_HBN_linux/bin/activate

# ========== CONFIGURATION ==========
# Choose encoder type: "autoencoder" or "crl"
ENCODER_TYPE="${ENCODER_TYPE:-autoencoder}"  # Default to autoencoder
echo "Using encoder type: $ENCODER_TYPE"

echo -e "\n1. Create adapted datasets for training"
python -m src.database_to_dataset.database_to_pretraining_shards # need to remove the limitation to SuS task (previous challenge rules, not usefull anymore)
python -m src.database_to_dataset.database_to_posttraining_shards # not used anymore (was first step for classif)
python -m src.database_to_dataset.database_to_challenge1_shards # specific to challenge 1 response time & classif (all on CDD task)

# ========== PRETRAINING ==========
if [ "$ENCODER_TYPE" = "crl" ]; then
    echo -e "\n1.CRL. Create multi-task CRL pretraining shards (if not already created)"
    # Uncomment to create CRL shards:
    # python -m src.database_to_dataset.database_to_crl_pretraining_shards --verbose --subjects-per-shard 25

    echo -e "\n2.CRL. Running CRL pretraining (200 epochs)..."
    python -m src.ML_pipeline_test.crl_pretraining --epochs 200 --batch-size 256
else
    echo -e "\n2. Running autoencoder pretraining (100 epochs)..."
    python -m src.ML_pipeline_test.pretraining --epochs 100 --batch-size 32
fi

echo -e "\n3. Running classification (50 epochs)..."
python -m src.ML_pipeline_test.classification --epochs 50 --batch-size 16 # not usefull for the real challenge now that they changed the rules

# ========== DOWNSTREAM REGRESSION ==========
echo -e "\n4.1 Running Regression - Response Time (50 epochs)..."
# To run regression on response time:
# In ML_pipeline_test/config.py, change DOWNSTREAM_DATA_PATTERN to "challenge1_data_shard_*.pkl" (and comment line 47)
python -m src.ML_pipeline_test.regression --encoder_type $ENCODER_TYPE --epochs 50 --batch-size 16 --target response_time
# Oo broken !!! # check again wich data is used and

# for the following regressions, in config.py, change DOWNSTREAM_DATA_PATTERN to "posttraining_data_shard_*.pkl" (and uncomment line 43)
echo -e "\n4.2 Running Regression - P-Factor (50 epochs)..."
python -m src.ML_pipeline_test.regression --encoder_type $ENCODER_TYPE --epochs 50 --batch-size 16 --target p_factor

echo -e "\n4.3 Running Regression - Attention (50 epochs)..."
python -m src.ML_pipeline_test.regression --encoder_type $ENCODER_TYPE --epochs 50 --batch-size 16 --target attention

echo -e "\n4.4 Running Regression - Internalizing (50 epochs)..."
python -m src.ML_pipeline_test.regression --encoder_type $ENCODER_TYPE --epochs 50 --batch-size 16 --target internalizing

echo -e "\n4.5 Running Regression - Externalizing (50 epochs)..."
python -m src.ML_pipeline_test.regression --encoder_type $ENCODER_TYPE --epochs 50 --batch-size 16 --target externalizing

# echo -e "\n5. Evaluating models..." # Need to rework the evaluation stuff
# first create the adapted evaluation dataset created using independant release of the database 
python -m src.ML_pipeline_test.evaluation.evaluation --create-datasets
python -m src.ML_pipeline_test.evaluation.evaluation --output-dir /home/mts/HBN_EEG_v1/src/ML_pipeline_test/saved_models

echo -e "\n========================================="
echo "Pipeline Complete!"
echo "Encoder type used: $ENCODER_TYPE"
echo ""
echo "To switch encoder type, run with:"
echo "  ENCODER_TYPE=crl ./run_pipeline.sh"
echo "  ENCODER_TYPE=autoencoder ./run_pipeline.sh"
echo "========================================="
