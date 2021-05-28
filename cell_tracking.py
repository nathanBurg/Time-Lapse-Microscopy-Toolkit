import numpy as np
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import trackpy as tp
import os
import cv2
import random2
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from scipy import ndimage
import pandas as pd
from pandas import DataFrame, Series  # for convenience
from wtershed_fxn import *


# This function opens the images in a folder
def open_folder(folder):
    images = []
    for filename in sorted(os.listdir(folder)):
        img = cv2.imread(os.path.join(folder,filename))
        if img is not None:
            images.append(img)
    return images

def get_args():
    parser = argparse.ArgumentParser(description='Count and track cells',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--folder', '-f', help='path to image folder', type=str, required=True)

    return parser.parse_args()


def cell_tracker(folder_pth):

    folder = open_folder(folder_pth)

    frames = []
    for i in range(len(folder)):
        ctr = wtershed(folder[i])
        frames.append(ctr)


    test_ind = np.random.randint(len(folder))
    #test_ind = 0
    f = tp.locate(frames[test_ind], 9, invert=False, separation=1,  minmass=300) # Identify and label every object
    plt.figure()
    tp.annotate(f, frames[test_ind], plot_style={'markersize': 1});
    #print(f.head())

    f = tp.batch(frames[:len(frames)], 5, invert=False, separation=1); # Label the whole batch


    pred = tp.predict.NearestVelocityPredict()
    search_range = 21
    memory = 0
    adaptive_stop = 0.5
    neighbor_strategy = 'BTree' # 'KDTree', 'BTree'
    link_strategy = 'numba'    # ‘recursive’, ‘nonrecursive’, ‘numba’, ‘drop’, ‘auto’
    t = tp.link_df(f, search_range,
                   adaptive_stop=adaptive_stop,
                   memory=memory
                   #neighbor_strategy=neighbor_strategy,
                   #link_strategy=link_strategy
                   ) # Link features into trajectories


    # 18 - 20 pix for search dist
    print(t.head())

    tracks = t.drop(['mass', 'size', 'ecc', 'signal', 'raw_mass', 'ep'], axis='columns')
    print(tracks.head())
    #print('Number of particles: ' + str(len(np.unique(tracks['particles']))))


    plt.figure()
    tp.plot_traj(t) #superimpose=folder[0]
    plt.show()

    frame_len = tracks['frame']


    cell_frame_cnt = dict()
    for fr in range(len(np.unique(frame_len))):
        wrd = 'Frame ' + str(fr)
        cell_cnt = np.sum(tracks['frame'] == fr)
        cell_frame_cnt[wrd] = cell_cnt

    cell_cnt_val = cell_frame_cnt.values()
    cell_cnt_ls = list(cell_cnt_val)
    cell_count_arr = np.array(cell_cnt_ls)
    cell_cnt_mean = np.mean(cell_count_arr)

    print("Mean number of cells per frame: " + str(cell_cnt_mean))


    part_ls =  list(tracks['particle'])

    track_dict = dict()
    for un in range(len(np.unique(part_ls))):
        word = 'particle ' + str(un)
        track_dict[word] = part_ls.count(un)

    track_dict_val = track_dict.values()
    track_ls = list(track_dict_val)
    track_arr = np.array(track_ls)

    num_particles = len(np.unique(part_ls))
    track_mean = np.mean(track_arr)

    print("Number of total tracks: " + str(num_particles))
    print("Mean track distance: " + str(track_mean))

    return tracks



if __name__ == '__main__':
    args = get_args()
    folder_pth = args.folder
    cell_tracking = cell_tracker(folder_pth=folder_pth)
