#!/bin/bash

echo "========================================="
echo "PyTorch EEG ML Pipeline - Complete Test"
echo "========================================="

source venv_HBN_linux/bin/activate

echo -e "\n1. Create adapted datasets for training"
python -m src.database_to_dataset.database_to_pretraining_shards # need to remove the limitation to SuS task (previous challenge rules, not usefull anymore)
python -m src.database_to_dataset.database_to_posttraining_shards # not used anymore (was first step for classif)
python -m src.database_to_dataset.database_to_challenge1_shards # specific to challenge 1 response time & classif (all on CDD task)

echo -e "\n2. Running pretraining (100 epochs)..."
python -m src.ML_pipeline_test.pretraining --epochs 100 --batch-size 32

echo -e "\n3. Running classification (50 epochs)..."
python -m src.ML_pipeline_test.classification --epochs 50 --batch-size 16 # not usefull for the real challenge now that they changed the rules

echo -e "\n4.1 Running Regression (50 epochs)..."
# To run regression on response time:
# In ML_pipeline_test/config.py, change DOWNSTREAM_DATA_PATTERN to "challenge1_data_shard_*.pkl" (and comment line 47)
python -m src.ML_pipeline_test.regression --epochs 50 --batch-size 16 --target response_time 
# Oo broken !!! # check again wich data is used and 

# for the following regressions, in config.py, change DOWNSTREAM_DATA_PATTERN to "posttraining_data_shard_*.pkl" (and uncomment line 43)
echo -e "\n4.2 Running Regression (50 epochs)..."
python -m src.ML_pipeline_test.regression --epochs 50 --batch-size 16 --target p_factor

echo -e "\n4.3 Running Regression (50 epochs)..."
python -m src.ML_pipeline_test.regression --epochs 50 --batch-size 16 --target attention

echo -e "\n4.4 Running Regression (50 epochs)..."
python -m src.ML_pipeline_test.regression --epochs 50 --batch-size 16 --target internalizing

echo -e "\n4.5 Running Regression (50 epochs)..."
python -m src.ML_pipeline_test.regression --epochs 50 --batch-size 16 --target externalizing

# echo -e "\n5. Evaluating models..." # Need to rework the evaluation stuff
# first create the adapted evaluation dataset created using independant release of the database 
python -m src.ML_pipeline_test.evaluation.evaluation --create-datasets
python -m src.ML_pipeline_test.evaluation.evaluation --output-dir /home/mts/HBN_EEG_Analysis/src/ML_pipeline_test/saved_models

echo -e "\n========================================="
echo "Pipeline Complete!"
echo "========================================="
