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

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n] # yields a list of size n

def create_tuple(list_, name1, name2):
    return [(l, name1, name2) for l in list_]

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def getKeyFromFiles(files):
    keys = []
    for f in files:
        items = f.split('/')[-1].split('_')
        zoom, x, y = items[-3], items[-2], items[-1][:-4]
        key = '_'.join((zoom, x, y))
        if key not in keys:
            keys.append(key)
    print(len(keys))

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
    mask = np.where(img_arr_cp  >= 0.3)
    no_mask = np.where(img_arr_cp < 0.3) # see how to do no mask
    img_arr[mask] = 1
    img_arr[no_mask] = 0
    print(np.unique(img_arr))
    return img_arr

def open_image(path):
    img_arr = cv2.imread(path)
    if img_arr is not None:
        return cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)

def search_hansen_file(name, files):
    for file in files:
        file_name = file.split('/')[-1]
        if file_name == name:
            return file
    return None

def create_forest_tile(forest_cover, hansen_loss, hansen_gain, out_dir, year, z, x, y):
    img_name = 'fc{year}_{z}_{x}_{y}.npy'
    img_name = img_name.format(year=year, z=z, x=x, y=y)
    loss_mask = np.where(hansen_loss != 0) # set to 0 wherever there is hansen loss
    img_arr = np.copy(forest_cover)
    if hansen_gain is not None:
        gain_mask = np.where(hansen_gain != 0) # set to 1 wherever there is hansen gain
        img_arr[gain_mask] = 1
    # Since the hansen gain has happened from 2000-2012 and the hansen loss from 2000-2016/18
    # We assume that the loss has happened after the hansen gain
    # So in case there is forest loss and forest gain, we assume that there has been
    # forest loss
    img_arr[loss_mask] = 0 # set to 0 wherever there has been loss
    np.save(os.path.join(out_dir, img_name), img_arr)
    logger.debug('SUCCESS: {}'.format(img_name))

    def create_forest_cover_tiles(forest_files, loss_files, gain_files, out_dir):
    loss_template = 'ly{year}_{z}_{x}_{y}.png'
    gain_template = 'fg2012_{z}_{x}_{y}.png'
    for file in forest_files:
        print('processing', file)
        forest_cover = open_image(file)
        forest_cover = process_forest_tile(forest_cover) # >= 0.3 is forest, < 0.3 is not
        year, z, x, y = get_tile_info(file)
        forest_gain = search_hansen_file
        hansen16 = search_hansen_file(loss_template.format(year='2016', z=z, x=x, y=y), loss_files)
        hansen17 = search_hansen_file(loss_template.format(year='2017', z=z, x=x, y=y), loss_files)
        hansen18 = search_hansen_file(loss_template.format(year='2018', z=z, x=x, y=y), loss_files)
        gain00 = search_hansen_file(gain_template.format(z=z, x=x, y=y))

        hansen_gain = open_image(gain00)
        if hansen16:
            hansen_loss16 = open_image(hansen16)
            create_forest_tile(forest_cover, hansen_loss16, hansen_gain, out_dir, '2016', z, x, y)
        if hansen17:
            hansen_loss17 = open_image(hansen17)
            create_forest_tile(forest_cover, hansen_loss17, hansen_gain, out_dir, '2017', z, x, y)
        if hansen18:
            hansen_loss18 = open_image(hansen18)
            create_forest_tile(forest_cover, hansen_loss18, hansen_gain, out_dir, '2018', z, x, y)

def main():
    source_dir = '/mnt/ds3lab-scratch/lming/data/min_quality'
    forest_cover2000_dir = os.path.join(source_dir, 'forest_cover_raw')
    forest_gain_2012_dir = os.path.join(source_dir, 'forest_gain')
    hansen_dir = os.path.join(source_dir, 'hansen')
    hansen_other = os.path.join(source_dir, 'hansen_other')

    out_forest_dir = os.path.join(source_dir, 'forest_cover_processed')
    create_dir(out_forest_dir)

    # forest loss
    forest_loss_files = getListOfFiles(hansen_dir)
    forest_loss_files_other = getListOfFiles(hansen_other)
    forest_loss_files.extend(forest_loss_files_other)

    # forest cover
    forest_files = getListOfFiles(forest_dir)

    # forest gain
    forest_gain_files = getListOfFiles(forest_gain_2012_dir)

    create_forest_cover_tiles(forest_files, forest_loss_files,
        forest_gain_files, out_forest_dir)

if __name__ == '__main__':
    main()
