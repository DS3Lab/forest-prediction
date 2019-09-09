"""Module to download "high quality" planet tiles (min 5% forest loss)
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

HANSEN18_URL = 'https://storage.googleapis.com/earthenginepartners-hansen/tiles/gfc_v1.6/loss_alpha/{z}/{x}/{y}.png'
HANSEN17_URL = 'https://storage.googleapis.com/earthenginepartners-hansen/tiles/gfc_v1.5/loss_alpha/{z}/{x}/{y}.png'
HANSEN16_URL = 'https://storage.googleapis.com/earthenginepartners-hansen/tiles/gfc_v1.4/loss_alpha/{z}/{x}/{y}.png'

logger = logging.getLogger('create_loss_tiles')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('create_loss_tiles.log')
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

def download_item(url, folder, item_type='hansen'):
    """
    Download quad tif
    """
    local_filename = '_'.join(url.split('/')[5:])
    version = local_filename[4:8]
    if version == 'v1.4':
        prefix = 'ly2016_'
    elif version == 'v1.5':
        prefix = 'ly2017_'
    else:
        prefix = 'ly2018_'
    local_filename = prefix + '_'.join(local_filename.split('_')[-3:])

    if os.path.isfile(os.path.join(folder, local_filename)):
        return os.path.join(folder, local_filename)
    # NOTE the stream=True parameter below
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            if folder:
                path = os.path.join(folder, local_filename)
            else:
                path = local_filename
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        # f.flush()
    except:
        logger.debug('FAILED: ' + url)
        return None
    logger.debug('DOWNLOADED: ' + url)

def get_tile_info(file):
    items = file.split('/')[-1].split('_')
    year, zoom, x, y, cx, cy = items[0][2:], items[1], items[2], items[3], items[4], items[5][:-4]
    return year, zoom, x, y, cx, cy

def need_file(files, year, zoom, x, y):
    template = 'ly{year}_{zoom}_{x}_{y}.png'
    fs = template.format(year=year, zoom=zoom, x=x, y=y)
    for file in files:
        f = file.split('/')[-1]
        if f == fs:
            return False
    return True

def get_file(files, year, zoom, x, y, cx, cy):
    template = 'ly{year}_{zoom}_{x}_{y}_{cx}_{cy}.png'
    fs = template.format(year=year, zoom=zoom, x=x, y=y, cx, cy)
    for file in files:
        f = file.split('/')[-1]
        if f == fs:
            return file
    return None

def get_url(year, zoom, x, y):
    if year == '2016':
        return HANSEN16_URL.format(z=zoom, x=x, y=y)
    elif year == '2017':
        return HANSEN17_URL.format(z=zoom, x=x, y=y)
    else:
        return HANSEN18_URL.format(z=zoom, x=x, y=y)
        logging.debug('WARNING, RETURNING HANSEN18', HANSEN18_URL.format(z=zoom, x=x, y=y))

def get_other_urls(hansen_files):
    other_urls = []
    for file in hansen_files:
        year, zoom, x, y, cx, cy = get_tile_info(file)
        year_check = str(int(year) - 1)
        if need_file(hansen_files, year_check, zoom, x, y):
            other_urls.append(get_url(year_check, zoom, x, y))
    return other_urls

# def get_forest_urls(hansen_files):
#     forest_urls = []
#     for file in hansen_files:
#         year, zoom, x, y = get_tile_info(file)
#         url = FOREST_URL.format(z=zoom,x=x,y=y)
#         forest_urls.append(url)
#     return forest_urls

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

# def getKeyFromFiles(files):
#     keys = []
#     for f in files:
#         items = f.split('/')[-1].split('_')
#         zoom, x, y = items[-3], items[-2], items[-1][:-4]
#         key = '_'.join((zoom, x, y))
#         if key not in keys:
#             keys.append(key)
#     print(len(keys))

def save_loss_tile(path1, path2, out_file):
    img_arr1 = cv2.imread(path1)
    img_arr2 = cv2.imread(path2)
    # print(path1, path2)
#     print(img_arr1.shape, img_arr2.shape)
    img_arr1 = cv2.cvtColor(img_arr1, cv2.COLOR_BGR2GRAY)
    img_arr2 = cv2.cvtColor(img_arr2, cv2.COLOR_BGR2GRAY)
    img = img_arr1 - img_arr2
    np.save(out_file, img.astype(np.uint8))

def create_loss_tiles(hansen_files, hansen_dir, out_dir):
    template = 'ly{year}_{zoom}_{x}_{y}_{cx}_{cy}.png'
    out_file_temp = 'ly{year}_{zoom}_{x}_{y}_{cx}_{cy}.npy'
    for file in hansen_files:
        year, zoom, x, y, cx_, cy = get_tile_info(file)
        if year in ['2017', '2018']:
            py_file = get_file(hansen_files, int(year)-1, zoom, x, y, cx, cy)
                if not py_file:
                    print(py_file, 'DOES NOT EXIST')
                    logger.debug('WARNING: {} doesnt exist'.format(py_file))
                    continue
            # out_file = out_file_temp.format(year=year, zoom=zoom, x=x, y=y)
            # save_loss_tile(file, py_file, os.path.join(out_dir, out_file))
            # logger.debug('file1 {}\nfile2{}\noutput{}'.format(file, py_file, out_file))

def main():
    ########### DOWNLOAD TILES ###############
    # out_dir = '/mnt/ds3lab-scratch/lming/data/min_quality'
    # out_hansen_dir = os.path.join(out_dir, 'hansen_other')
    # out_hansen_loss_dir = os.path.join(out_dir, 'hansen_loss')
    # hansen_dir = os.path.join(out_dir, 'hansen')
    # create_dir(out_hansen_dir)
    # create_dir(out_hansen_loss_dir)

    hansen_cropped_dir = '/mnt/ds3lab-scratch/lming/data/min_quality/forest_loss_raw_cropped'
    # Get hansen files
    hansen_files = getListOfFiles(hansen_dir)
    # other_urls = get_other_urls(hansen_files)
    # print(len(other_urls))
    # import pickle as pkl
    # with open('other_urls.pkl', 'rb') as pkl_file:
    #     other_urls = pkl.load(pkl_file)
    #
    # for chunk in chunks(other_urls, 16):
    #     with multiprocessing.Pool(processes=16) as pool:
    #         results = pool.starmap(download_item, create_tuple(chunk, out_hansen_dir, 'hansen'))

    ########### SAVE LOSS TILES ###############
    create_loss_tiles(hansen_files, , hansen_dir, out_hansen_loss_dir)

if __name__ == '__main__':
    main()
