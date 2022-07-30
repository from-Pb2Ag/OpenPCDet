#Read split files and append frame ids to a list
#shuffle indices of len of list
#extract 60% of the list to train, 20% to val, 20% to test

from pathlib import Path
import numpy as np
from tqdm import tqdm
import random
random.seed(100)

ROOT_PATH = Path('/home/barza/OpenPCDet/data/dense')
original_splits_path = ROOT_PATH / 'ImageSets' / 'original_splits'
splits = ['clear', 'snow', 'dense_fog', 'light_fog']

split_percentages = [('train', 60), ('val', 15), ('test', 25)]

for split in splits:
    split_path = original_splits_path / f'{split}_FOV3000.txt'
    split_ids = [x.strip() for x in open(split_path).readlines()] 
    print(f'Read {split_path} with {len(split_ids)} samples')

    num_frames = len(split_ids)
    shuffled_indices = list(range(num_frames))
    random.shuffle(shuffled_indices)

    start_idx = 0
    for s,p in split_percentages:
        num_idx_select = int(p*num_frames/100)
        assert start_idx + num_idx_select < num_frames
        idx_selected = shuffled_indices[start_idx:start_idx+num_idx_select]
        start_idx = start_idx+num_idx_select
        new_split_path = original_splits_path / f'{s}_{split}_FOV3000_{p}.txt'
        print(f'Writing in {new_split_path} {len(idx_selected)} samples')
        with open(new_split_path, 'w') as f:
            for i, idx in enumerate(idx_selected):
                f.write(split_ids[idx])
                if i != len(idx_selected)-1 :
                    f.write('\n')