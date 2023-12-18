from datetime import datetime

import sys
import time
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from racecar_gym.core.gridmaps import GridMap


def pos_checkd(data_root, scenario, save_img_dir, pos_count):
    # Get all dir in DATA_ROOT
    pos_files = [os.path.join(data_root, f) for f in os.listdir(data_root)]
    img_files = [os.path.join(save_img_dir, f) for f in os.listdir(save_img_dir)]
    num_files = len(pos_files)
    num_image = len(img_files)
    
    

    chunk = 100
    npos = 20
    num = num_files - (num_image*100)


    if num < chunk: return pos_count
    
    print("totalf:", num_files, "total_img:", num_image)
    if num > chunk:
        sort_pos_files = sorted(pos_files)
        n_pos_files = [sort_pos_files[i+chunk-npos:i+chunk] for i in range((num_image*100), len(sort_pos_files), chunk)]
    else:
        sort_pos_files = sorted(pos_files)[-npos:]
        n_pos_files = [sort_pos_files]
    
    # Take the last LAST_N files (each file has a distance of FREQ)
    

    if "circle_cw" in scenario:
        map_path = 'racecar_gym_competition_env/models/scenes/circle_cw_competition/maps/maps.npz'
        MAP = np.load(map_path)
        ORIGIN = (-50, -50, 0.0)
        GRID = GridMap(grid_map=MAP['drivable_area'],
                       origin=ORIGIN,
                       resolution=0.05)
        CROP = (slice(950, 1350), slice(850, 1250))
    elif "austria" in scenario:
        map_path = 'racecar_gym_competition_env/models/scenes/austria_competition/maps/maps.npz'
        MAP = np.load(map_path)
        ORIGIN = (-50, -50, 0.0)
        GRID = GridMap(grid_map=MAP['drivable_area'],
                       origin=ORIGIN,
                       resolution=0.05)
        CROP = (slice(950, 1450), slice(850, 1350))
    else:
        raise NotImplementedError

    

    alpha_=1./(npos/4)

    nn = 0
    for pos_files in n_pos_files:
        if len(pos_files) != npos: break
        print(nn)
        nn+=1
        plt.imshow(MAP['drivable_area'][CROP[0], CROP[1]], cmap='gray')
        for pos_file in tqdm(pos_files):
            # Load infos
            infos = np.load(pos_file)

            # Load positions
            positions = np.array([info for info in infos])

            # Get
            xs = [GRID.to_pixel(positions[i][:2])[1] for i in range(len(positions)) if i%2 == 0]
            ys = [GRID.to_pixel(positions[i][:2])[0] for i in range(len(positions)) if i%2 == 0]

            # Crop
            xs_, ys_ = [], []
            for x, y in zip(xs, ys):
                # if CROP[0].start <= x < CROP[0].stop and CROP[1].start <= y < CROP[1].stop:
                xs_.append(x - CROP[1].start)
                ys_.append(y - CROP[0].start)

            xs, ys = xs_, ys_

            assert len(xs) == len(ys)
            for i in range(len(xs)):
                # plt.scatter(xs[i], ys[i], color='red', s=0.1, alpha=0.01)
                plt.scatter(xs[i], ys[i], color='red', s=0.1, alpha=alpha_)


        time_str = datetime.now().strftime('%Y%m%d-%H:%M:%S')
        plt.savefig(os.path.join(save_img_dir, time_str))
        plt.close()
    
    return pos_count+nn*chunk

if __name__ == '__main__':
    if len(sys.argv) !=2:
        print("Usage: python pos_check.py [folder name]")
        exit(1)

    folder_name=sys.argv[1]

    data_root = 'record/'+folder_name
    save_dir = data_root+'_img/'
    comp_map = "circle_cw" if folder_name[0]=='c' else "austria"
    
    Path(save_dir).mkdir(exist_ok=True, parents=True)
    begin=0
    while(True):
        print(begin)
        newb = pos_checkd(data_root, comp_map, save_dir, begin)
        if begin == newb: time.sleep(180)

        


