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
    return cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)

def search_hansen_file(name, files):
    for file in files:
        file_name = file.split('/')[-1]
        if file_name == name:
            return file
    return None

def create_forest_tile(forest_cover, hansen_loss, out_dir, year, z, x, y):
    img_name = 'fc{year}_{z}_{x}_{y}.npy'
    img_name = img_name.format(year=year, z=z, x=x, y=y)
    mask = np.where(hansen_loss != 0) # set to 0 wherever there is hansen loss
    img_arr = np.copy(forest_cover)
    img_arr[mask] = 0 # set to 0 where there has been loss
    np.save(os.path.join(out_dir, img_name), img_arr)
    logger.debug('SUCCESS: {}'.format(img_name))

def create_forest_cover_tiles(forest_files, hansen_files, out_dir):
    hansen_template = 'ly{year}_{z}_{x}_{y}.png'
    for file in forest_files:
        print('processing', file)
        forest_cover = open_image(file)
        forest_cover = process_forest_tile(forest_cover)
        year, z, x, y = get_tile_info(file)
        hansen16 = search_hansen_file(hansen_template.format(year='2016', z=z, x=x, y=y), hansen_files)
        hansen17 = search_hansen_file(hansen_template.format(year='2017', z=z, x=x, y=y), hansen_files)
        hansen18 = search_hansen_file(hansen_template.format(year='2018', z=z, x=x, y=y), hansen_files)
        if hansen16:
            hansen_loss16 = open_image(hansen16)
            create_forest_tile(forest_cover, hansen_loss16, out_dir, '2016', z, x, y)
        if hansen17:
            hansen_loss17 = open_image(hansen17)
            create_forest_tile(forest_cover, hansen_loss17, out_dir, '2017', z, x, y)
        if hansen18:
            hansen_loss18 = open_image(hansen18)
            create_forest_tile(forest_cover, hansen_loss18, out_dir, '2018', z, x, y)

def main():
    source_dir = '/mnt/ds3lab-scratch/lming/data/min_quality'
    forest_dir = os.path.join(source_dir, 'forest')
    out_forest_dir = os.path.join(source_dir, 'forest_cover')
    hansen_dir = os.path.join(source_dir, 'hansen')
    hansen_other = os.path.join(source_dir, 'hansen_other')
    create_dir(out_forest_dir)
    hansen_files = getListOfFiles(hansen_dir)
    hansen_files_other = getListOfFiles(hansen_other)
    hansen_files.extend(hansen_files_other)
    forest_files = getListOfFiles(forest_dir)

    create_forest_cover_tiles(forest_files, hansen_files, out_forest_dir)

if __name__ == '__main__':
    main()
