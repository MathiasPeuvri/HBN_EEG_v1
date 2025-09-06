# HBN EEG Analysis Project Overview

Pipeline for EEG data from the HBN datase

## Project Architecture

```
--- what is not shared ---
database/
├── R1_L100/                # all data from release 1 
├── R2_mini_L100_bdf/       # small release R2
datasets/
├── evaluation_datasets/    # subfolder with datasets dedicated to evaluation
├── *.pkl                   # all datasets files created
docs/
├── *.md                    # too long and verbose AI generated docs files ... to be deleted prbably
├── documentation/          # folder with word files, pictures and others (shared directly in rar mail ?)
legacy/                     # many scripts that have been removed because not working well / useless


scripts_task_vizualisation/
├── utils and script for each behavioral task in database
src/
├── database_downloader/         # AWS S3 dataset download utilities
│   ├── downloader.py               # S3 download logic with progress tracking
│   └── download_release2mini.py    # script to download R2 mini, could update to download the full database (->Main download script)
│
├── loader/                      #  EEG data loading
│   ├── simple_loader.py            # Core SimpleHBNLoader class
│   └── config.py                   # Configuration management
│
├── preprocessing/               # EEG signal processing pipeline  
│   ├── filters.py               # Filtering/normalizing the 'raw' data
│   ├── test_**.py               # scripts to check the preprocessing steps
│   └── epoching.py              # Epoching/splitting the time serie in small samples (2sec for ex)
│
├── database_to_dataset/         # Convert raw database → ML-ready datasets
│   ├── database_to_dataset.py      # Initial conversion logic (from raw continuous EEG rec to dataframe (stored .pkl) of EEG samples)
│   └── database_to_**_shards.py    # Create multiple .pkl for the different parts of the machine learning pipeline
│
└── ML_pipeline_test/            # Complete PyTorch ML training pipeline
    ├── config.py                    # Central configuration hub 
    ├── datasets_loader_classes/     # from .pkl files to pytorch DataLoader for different tasks
    │   ├── ssl_dataset.py              # Initial simple loader (Output only Pytorch ready EEG signals)
    │   ├── downstream_dataset.py       # Initial Task-specific downstream dataset (Gives Pytorch ready EEG signals & some task value)
    │   ├── shard_pretraining_dataset.py    # Sharded pretraining data
    │   └── shard_downstream_dataset.py     # Sharded downstream data
    │    
    ├── models.py                    # Simple model architecture and masking functions (should be split in two diff scripts)
    ├── data_loader_ml.py            # link the dataset_loader_classes to the Pytorch model according to the dataset_type ('pretraining', 'classification', or 'regression')
    ├── pretraining.py               # Self supervised pretraining using the masking approach
    ├── classification.py            # Downstream classification - don't work well but not used anymore
    ├── regression.py                # Downstream regression pipeline 
    ├── utils.py                     # Helper functions and metrics (/holder/not that good functions)
    ├── run_pipeline.sh              # Basically how to run the ML pipeline
    │
    ├── evaluation/                  # Model evaluation framework # all require carefull reviewing
    │   ├── eval_data_prep.py           # database_to_dataset wrapper to prepare evaluation data (on different release compared to the trainings)
    │   ├── evaluation.py               # Main evaluation logic
    │   └── metrics.py                  # Performance metrics calculation
    │
    └── saved_models/                # Model checkpoints
        ├── autoencoder_best.pth        # Best pretrained autoencoder
        ├── classifier_best.pth         # from when classification was part of the challenge
        ├── regressor_**_best.pth       # Best pretrained autoencoder
        └── evaluation_results.csv      # Output of the evaluation script (shows how bad the model is for now)
```

## Core Components

### Data Flow Pipeline
```
Raw HBN (Database) → raw EEG → Preprocessing/Epochs → Datasets → ML pipeline (load specific dataset / create model / train / eval)
     ↓                 ↓                ↓                 ↓                           ↓
 S3 Download        loader/       preprocessing/     database_to_dataset/        ML_pipeline_test/
```

### Key Modules

**scripts_task_vizualisation** (outside of src)
- some test scripts that try to create plot to show each task and an EEG trace in parallel

**database_downloader** 
- Pretty much ready to download full database (either 100 or 500Hz, mini or full release); just require to loop the download_release2mini (and remove the mini req)

**Loader Module**
- Simple version (have an enhance but extensive and complex in legacy) that just works well

**Preprocessing Module**: basic signal processing
- Bandpass filtering / notch / normalization (maybe still have multiple versions, comparing MNE vs custom scipy based)
- Epoching with configurable window sizes (and can epoch according to events)


                    # -- where stuff starts to be messy. -- #

**database_to_dataset**:
- Basic : load raw data; preprocess, split in desired epochs, return dataframe (.pkl) for data_loader_ml. Now use shards to split large database in multiple smaller datasets.
- database_to_pretraining : - split all SuS data from release 1 into dataset. - Should be challenge 1 decodeur pretraining - Also have the subject ID in the dataset to get back patient factors in order to run as data for challenge 2 (but it's kinda wrong for the real challenge)
-database_to_challenge1 : Split CCD data to get 2 sec pre-target EEG data and related reaction time.
- database_to_posttraining_shards : split the data according to comportemental task metadata (task_name: str = "contrastChangeDetection",
    target_events: List[str] = ['right_target', 'right_buttonPress', 'left_target', 'left_buttonPress'])  -- was planned for classification, not used anymore --

**datasets_loader_classes**: from .pkl to X / Y Pytorch compatible data 
- non-shards are 'legacy'
- shard_pretraining : just return the segemented EEG data (all shards from selected folder (see ML_pipeline_test config))
- shard_downstream : find if used for classif or regression, and adapt the Y for the downstream task

**core ML_pip_test**: configs, simple model and training pipelines
- configs: control the data used for ml pipeline and other ml params (standard/AI decided for now)
- data_loader_ml : adapt the call of dataset_loader_classes for the ml pipeline
- model : need reorga; very simple 3-layer encoder/decoder + binary classif head + masking strategy ... (Masking (bad, just mask random points, not more specific portions/channels) could be put in new file for self-supervised learning starts. Binnary classif should be moved away (not used anymore))
- pretraining : epochs for self supervised pre-training
- regression : regression head + epochs for regression training

(- utils : make some validation and compute metrics - not really checked (AI implemented for now)
- classification : don't work well - not used anymore)

*run_pipeline.sh* : run a pretraining then 5 regression for each regression task model.

**evaluation** !! AI generated and not reviewd !!
- eval_data_prep : call the database_to_dataset to create isolated evaluation datasets
- evaluation : eval pipeline; load pretrained encodeur/regression head; run on eval_dataset and create metric outputs csv
- metrics : compute some metrics for evaluation (I think it just output some mean over all outputs from the evaluation run)
