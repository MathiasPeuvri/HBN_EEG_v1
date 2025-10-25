"""
Tasks included:
- RestingState (RS), surroundSupp (SuS), contrastChangeDetection (CCD), 
seqLearning8target (SL), symbolSearch (SyS), DespicableMe (MW)
"""

import pickle
import argparse
from pathlib import Path
from typing import Optional, List
import pandas as pd
import warnings
import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
warnings.filterwarnings("ignore", category=RuntimeWarning)

from braindecode.preprocessing import create_fixed_length_windows
from braindecode.datasets.base import BaseConcatDataset
from eegdash import EEGChallengeDataset


# All 6 HBN tasks for multi-task pretraining
DEFAULT_TASKS = ["RestingState", "surroundSupp", "contrastChangeDetection",
 "seqLearning8target", "symbolSearch", "DespicableMe"]

releases = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10", "R11"]

releases = [ "R7", "R8", "R9", "R10", "R11"]
#releases = [ "R6", "R7", ]
from pathlib import Path

DATABASE_DIR = PROJECT_ROOT / "database"
print(f'Project root: {PROJECT_ROOT}, database directory: {DATABASE_DIR}')
from eegdash.dataset import EEGChallengeDataset
from joblib import Parallel, delayed
import time
for release in releases:
    for task in DEFAULT_TASKS:
        print(f'\n\nDownloading {task} from {release}\n\n')

        dataset_ccd = EEGChallengeDataset(task=task,
                                        release=release, cache_dir=DATABASE_DIR,
                                        mini=False)

        raws = Parallel(n_jobs=4)(delayed(lambda d: d.raw)(d) for d in dataset_ccd.datasets)

        time.sleep(2)