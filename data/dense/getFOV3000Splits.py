'''
Create new splits with frames that have FOV points more than 3000.
'''

from pathlib import Path
from pcdet.utils import calibration_kitti
import numpy as np
from tqdm import tqdm

ROOT_PATH = Path('/home/barza/OpenPCDet/data/dense')
LIDAR_FOLDER = Path('/media/barza/WD_BLACK') / 'datasets' / 'dense' / 'lidar_hdl64_strongest'
original_splits_path = ROOT_PATH / 'ImageSets' / 'original_splits'
splits = ['clear', 'snow', 'dense_fog', 'light_fog']

def get_calib(sensor: str = 'hdl64'):
    calib_file = ROOT_PATH / f'calib_{sensor}.txt'
    assert calib_file.exists(), f'{calib_file} not found'
    return calibration_kitti.Calibration(calib_file)


def get_fov_flag(pts_rect, img_shape, calib):

    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag


for split in splits:
    split_path = original_splits_path / f'{split}.txt'
    sample_id_list = [x.strip() for x in open(split_path).readlines()]
    new_sample_id_list = []
    frames_removed = 0

    print(f'Filtering {split}:')

    #Get new sample id list with frames more than 3000 points in FOV
    for sample_idx in tqdm(sample_id_list):
        # Get points
        lidar_sample_idx = '_'.join(sample_idx.split(','))
        lidar_file = LIDAR_FOLDER / f'{lidar_sample_idx}.bin'
        pc = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 5)
        
        calibration = get_calib()
        pts_rectified = calibration.lidar_to_rect(pc[:, 0:3])
        fov_flag = get_fov_flag(pts_rectified, (1024, 1920), calibration)
        pc_fov = pc[fov_flag]

        if pc_fov.shape[0] < 3000:
            print(sample_idx)
            frames_removed += 1
            continue
        
        new_sample_id_list.append(sample_idx)
    
    print(f'Frames removed: {frames_removed}, Total {split} frames: {len(sample_id_list)}')
    
    #Generate new <split>_FOV3000.txt
    dst_txt = original_splits_path / f'{split}_FOV3000.txt'
    with open(dst_txt, 'w') as f:
        for i, sample_idx in enumerate(new_sample_id_list):
            f.write(sample_idx)
            if i != len(new_sample_id_list) - 1:
                f.write('\n')


