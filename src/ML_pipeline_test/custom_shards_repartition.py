from pathlib import Path
import os
import numpy as np

epoch = 50
shards_nb = 10 # will be used after warmup

def custom_shards_repartition(data_dir: Path, epoch: int, shards_nb: int):
    all_shards = os.listdir(data_dir)
    # on hardcode le fait d'utiliser la release 2 pour la validation et la 5 pour le test
    val_shards = [f for f in os.listdir(data_dir) if f.endswith('_R2.pkl')]
    test_shards = [f for f in os.listdir(data_dir) if f.endswith('_R5.pkl')]
    train_shards = [f for f in all_shards if f not in val_shards and f not in test_shards]

    epoch_files_lists = [] # will containe one list of files for each epoch

    # first put full releases in epoch_files_lists
    releases_in_train = np.unique([f.split('.pkl')[0].split('_R')[-1:] for f in train_shards])
    import random
    rng = random.Random(42)#self.seed
    rng.shuffle(releases_in_train)
    full_releases_epoch = len(releases_in_train)
    for iepoch, full_release_epoch in enumerate(range(full_releases_epoch)):
        #print(f'epoch {iepoch+1} full release: {releases_in_train[iepoch]}')
        epoch_files = [data_dir / f for f in train_shards if f'_R{releases_in_train[iepoch]}.pkl' in f]        
        epoch_files_lists.append(epoch_files)

    # then shuffled shards for remaining epochs (shard_nb / epoch)
    shard_shuff = []
    remaining_epochs = epoch - full_releases_epoch
    for iepoch, remaining_epoch in enumerate(range(remaining_epochs)):
        if len(shard_shuff) < shards_nb:
            shard_shuff = train_shards.copy()
            rng.shuffle(shard_shuff)
        # take first shards_nb shards for this epoch
        epoch_files = [data_dir / f for f in shard_shuff[:shards_nb]]
        shard_shuff = shard_shuff[shards_nb:]
        epoch_files_lists.append(epoch_files)
    return epoch_files_lists

if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
    DATA_DIR = PROJECT_ROOT / "dataset_test"
    epoch = 50
    shards_nb = 10 # will be used after warmup
    epoch_files_lists = custom_shards_repartition(DATA_DIR, epoch, shards_nb)
    print(f"epoch_files_lists:")
    for e in epoch_files_lists:
        print(f"epoch_files:\n {e}\n")