import glob
from config import DOWNSTREAM_DATA_PATTERN
from config import DOWNSTREAM_CHALL1_CLICKCENTERED_PATTERN
from config import DOWNSTREAM_CHALL1_PATTERN

pattern = DOWNSTREAM_DATA_PATTERN
for pattern in [DOWNSTREAM_DATA_PATTERN, 
DOWNSTREAM_CHALL1_CLICKCENTERED_PATTERN,
DOWNSTREAM_CHALL1_PATTERN]:
    # from pattern to train / val pattern:
    all_patterns = glob.glob(pattern)
    print(all_patterns)
    train_pattern = [f for f in all_patterns if 'R2' not in f]
    val_pattern = [f for f in all_patterns if 'R2' in f]
    print("\ntrain_pattern: \n", train_pattern)
    print("\nval_pattern: \n", val_pattern)
