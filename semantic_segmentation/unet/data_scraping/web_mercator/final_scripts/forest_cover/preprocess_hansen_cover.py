"""Module to preprocess the hansen cover
It will
"""

import requests
import os
import sys
import json
import pickle as pkl
import multiprocessing
import glob
import re
import math
import logging
import cv2
import numpy as np
from itertools import product

logger = logging.getLogger('forest-preprocess')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('forest-preprocess.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_tile_info(file):
    file_name = file.split('/')[-1]
    name_split = file_name.split('_')
    year = name_split[0][2:]
    zoom = name_split[1]
    x = name_split[2]
    y = name_split[3][:-4]
    return year, zoom, x, y

def process_forest_tile(img_arr):
    img_arr_cp = np.copy(img_arr)
    img_arr_cp = img_arr_cp / 255.
    mask = np.where(img_arr_cp  >= 0.25)
    no_mask = np.where(img_arr_cp < 0.25) # see how to do no mask
    img_arr[mask] = 1
    img_arr[no_mask] = 0
    print(np.unique(img_arr))
    return img_arr

def open_image(path):
    img_arr = cv2.imread(path)
    if img_arr is not None:
        return cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)

def create_forest_cover_tile(fc_img, fg_img, fl_img, out_img_dir):
    img_arr = np.copy(fc_img)
    if fg_img is not None:
        gain_mask = np.where(fg_img != 0)
        img_arr[gain_mask] = 1
    if fl_img is not None:
        loss_mask = np.where(fl_img != 0) # set to 0 wherever there is hansen loss
        img_arr[loss_mask] = 0
    # Since the hansen gain has happened from 2000-2012 and the hansen loss from 2000-2016/18
    # We assume that the loss has happened after the hansen gain
    # So in case there is forest loss and forest gain, we assume that there has been
    # forest loss
    np.save(out_img_dir, img_arr)
    logger.debug('SUCCESS: {}'.format(out_img_dir))

def create_forest_cover_tiles(forest_cover_files, forest_loss_dir,
    forest_gain2012_dir, out_dir):

    loss_template = 'fl{year}_{z}_{x}_{y}.png'
    gain_template = 'fg2012_{z}_{x}_{y}.png'
    years = ['2013', '2014', '2015', '2016', '2017']
    out_img_template = 'fc{year}_{z}_{x}_{y}.png'

    for file in forest_cover_files:
        print('processing', file)
        fc_img = open_image(file)
        fc_img = process_forest_tile(fc_img) # >= 0.25 is forest, < 0.25- is not

        year, z, x, y = get_tile_info(file)

        fg_path = os.path.join(forest_gain2012_dir, gain_template.format(z=z, x=x, y=y))
        fg_img = open_image(fg_path)
        for year in years:
            fl_path = os.path.join(forest_loss_dir, year, loss_template.format(year=year, z=z, x=x, y=y))
            fl_img = open_image(fl_path)
            out_img_dir = os.path.join(out_dir, year, out_img_template.format(year=year, z=z, x=x, y=y))
            create_forest_cover_tile(fc_img, fg_img, fl_img, out_img_dir)
        break


def main():
    src_dir = '/mnt/ds3lab-scratch/lming/data/min_quality11'
    forest_cover2000_dir = os.path.join(src_dir, 'forest_cover', 'min_pct')
    forest_gain2012_dir = os.path.join(src_dir, 'forest_gain', 'min_pct')
    forest_loss_dir = os.path.join(src_dir, 'forest_loss', 'min_pct')

    out_dir = os.path.join(source_dir, 'forest_cover', 'processed')
    create_dir(out_dir)
    years = ['2013', '2014', '2015', '2016', '2017']
    for year in years:
        create_dir(os.path.join(out_dir, year))

    forest_cover_files = glob.glob(os.path.join(forest_cover2000_dir, '*.png'))

    create_forest_cover_tiles(forest_cover_files, forest_loss_dir,
        forest_gain2012_dir, out_dir)

if __name__ == '__main__':
    main()
