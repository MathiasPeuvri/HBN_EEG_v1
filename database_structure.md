# HBN EEG Database Structure

AI generated document that describes the expected structure of the HBN (Healthy Brain Network) EEG database. Use this as a reference to verify that your downloaded database matches the expected organization.

## Overview

The HBN EEG database contains neurophysiological recordings from multiple subjects, organized in BIDS (Brain Imaging Data Structure) format. The database includes two main releases with different formats:

- **R1_L100**: Primary database with .set format EEG files  
- **R2_mini_L100_bdf**: Smaller subset with .bdf format EEG files

## Root Database Structure

```
database/
├── R1_L100/                    # Main database (135+ subjects)
│   ├── code/                   # Processing scripts
│   ├── dataset_description.json
│   ├── participants.json       # Participant metadata schema
│   ├── participants.tsv        # Participant demographics
│   ├── README                  # Database documentation
│   └── sub-NDAR*/             # Subject folders
│
└── R2_mini_L100_bdf/          # Mini database (20 subjects)
    ├── code/
    ├── dataset_description.json
    ├── participants.json
    ├── participants.tsv
    ├── README
    └── sub-NDAR*/             # Subject folders
```

## Subject Directory Structure

Each subject has a consistent folder structure:

```
sub-NDAR[ID]/
└── eeg/                       # All EEG data for this subject
    └── [EEG files]
```

### Example Subject IDs

**R1_L100 subjects (sample):**
- sub-NDARAC904DMU
- sub-NDARAG143ARJ
- sub-NDARAM704GKZ
- sub-NDARAN385MDH
- sub-NDARAP359UM6
- sub-NDARAW320CGR
- sub-NDARBD879MBX

**R2_mini_L100_bdf subjects (sample):**
- sub-NDARAB793GL3
- sub-NDARAM675UR8
- sub-NDARBM839WR5
- sub-NDARBU730PN8
- sub-NDARCT974NAJ

## EEG Data Files

Each subject's eeg/ folder contains multiple recordings with standardized naming:

### File Naming Convention
```
sub-[SUBJECT_ID]_task-[TASK_NAME]_[run-N]_[FILETYPE].[EXT]
```

### File Types per Recording

**For R1_L100 (EEGLAB format):**
- `*_eeg.set` - EEG data file (EEGLAB format)
- `*_channels.tsv` - Channel information
- `*_eeg.json` - Recording metadata
- `*_events.tsv` - Event markers and timing

**For R2_mini_L100_bdf (BioSemi format):**
- `*_eeg.bdf` - EEG data file (BioSemi format)
- `*_channels.tsv` - Channel information
- `*_eeg_channels.tsv` - Additional channel details
- `*_eeg.json` - Recording metadata
- `*_events.tsv` - Event markers and timing

### Available Tasks

The database includes recordings from 9 different experimental tasks:

1. **contrastChangeDetection** (multiple runs: run-1, run-2, run-3)
2. **DespicableMe** (video watching)
3. **DiaryOfAWimpyKid** (video watching)
4. **FunwithFractals** (visual task)
5. **RestingState** (eyes open/closed)
6. **seqLearning8target** (sequence learning)
7. **surroundSupp** (surround suppression)
8. **symbolSearch** (cognitive task)
9. **ThePresent** (video watching)

### Example File Names

From subject sub-NDARAC904DMU:
```
sub-NDARAC904DMU_task-contrastChangeDetection_run-1_channels.tsv
sub-NDARAC904DMU_task-contrastChangeDetection_run-1_eeg.json
sub-NDARAC904DMU_task-contrastChangeDetection_run-1_eeg.set
sub-NDARAC904DMU_task-contrastChangeDetection_run-1_events.tsv
sub-NDARAC904DMU_task-DespicableMe_channels.tsv
sub-NDARAC904DMU_task-DespicableMe_eeg.json
sub-NDARAC904DMU_task-DespicableMe_eeg.set
sub-NDARAC904DMU_task-DespicableMe_events.tsv
```

## Processed Datasets Structure

Processed data shards are stored separately from the raw database:

```
datasets/
├── pretraining_data_shard_0.pkl    # Pretraining data shards
├── pretraining_data_shard_1.pkl
├── pretraining_data_shard_2.pkl
├── pretraining_data_shard_3.pkl
├── posttraining_data_shard_0.pkl   # Post-training data shards
├── posttraining_data_shard_1.pkl
├── posttraining_data_shard_2.pkl
├── posttraining_data_shard_3.pkl
├── challenge1_data_shard_0.pkl     # Challenge 1 data shards
├── challenge1_data_shard_1.pkl
├── challenge1_data_shard_2.pkl
├── challenge1_data_shard_3.pkl
├── challenge_1_data.pkl            # Original challenge data
└── evaluation_datasets/             # Evaluation-specific datasets
```

## Verification Checklist

Use this checklist to verify your database is correctly structured:

### Database Root
- [ ] `database/` folder exists
- [ ] `database/R1_L100/` folder exists (if using R1)
- [ ] `database/R2_mini_L100_bdf/` folder exists (if using R2)

### Database Metadata Files
- [ ] `dataset_description.json` present in each database
- [ ] `participants.tsv` present with subject demographics
- [ ] `participants.json` present with metadata schema
- [ ] `README` file present with documentation

### Subject Folders
- [ ] Subject folders follow pattern: `sub-NDAR[ID]`
- [ ] Each subject has an `eeg/` subfolder
- [ ] At least 20 subjects present (R2_mini) or 135+ subjects (R1)

### EEG Files per Subject
- [ ] Each recording has 4 associated files (data, channels, json, events)
- [ ] File extensions match database type (.set for R1, .bdf for R2)
- [ ] Task names are consistent across subjects
- [ ] Run numbers present for multi-run tasks (e.g., contrastChangeDetection)

### Processed Data (if applicable)
- [ ] `datasets/` folder exists
- [ ] Shard files follow naming pattern: `[stage]_data_shard_[N].pkl`
- [ ] Shards are numbered sequentially (0, 1, 2, 3)

## File Size Expectations

**Typical file sizes:**
- `.set` files: 5-30 MB per recording
- `.bdf` files: 10-50 MB per recording  
- Pretraining shards: 1.5-2.1 GB each
- Posttraining shards: 600-825 MB each
- Challenge1 shards: 280-380 MB each

## Notes

- The database follows BIDS specification for neurophysiology data
- Not all subjects have recordings for all tasks
- Some tasks have multiple runs, others have single recordings
- Event files contain trial timing and stimulus information
- Channel files specify electrode locations and properties