'''
Check if Martin's DROR folder has all lidar frames from our train_adverse_weather_60 split
'''
from pathlib import Path
from random import sample
import numpy as np
from tqdm import tqdm
import glob

DROR_PATH = '/media/barza/WD_BLACK/datasets/dense/DROR/alpha_0.45/all/hdl64/strongest/full'
DROR_samples = glob.glob(f'{DROR_PATH}/*.pkl')

root_split_path = Path('/home/barza/OpenPCDet/data/dense/ImageSets')
splits = ['train_dense_fog_60.txt', 'train_light_fog_60.txt', 'train_snow_60.txt']

dror_sample_list =[]
for sample_path in DROR_samples:
    dror_sample_idx = sample_path.split('/')[-1].split('.')[0]
    dror_sample_list.append(dror_sample_idx)

for split in splits:
    print(split)
    split_path = root_split_path / split 
    sample_id_list = ['_'.join(x.strip().split(',')) for x in open(split_path).readlines()]
    id_not_found = 0
    for id in tqdm(sample_id_list):
        if id not in dror_sample_list:
            print(f'Idx not found in DROR: {id}, split: {split}')
            id_not_found +=1
    print(f'Total ids not found: {id_not_found}, split len: {len(sample_id_list)}')


