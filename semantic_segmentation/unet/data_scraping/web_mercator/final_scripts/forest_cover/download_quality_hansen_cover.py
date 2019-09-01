"""Module to download high quality forest cover tiles
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

FOREST_URL = "https://storage.googleapis.com/earthenginepartners-hansen/tiles/gfc_v1.4/tree_gray/{z}/{x}/{y}.png"
FOREST_GAIN_URL = "https://storage.googleapis.com/earthenginepartners-hansen/tiles/gfc_v1.4/gain_alpha/{z}/{x}/{y}.png"

logger = logging.getLogger('forest')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('forest.log')
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
    prefix = 'fg2012_'
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

def get_forest_urls(hansen_files, URL):
    forest_urls = []
    for file in hansen_files:
        file_name = file.split('/')[-1]
        name_split = file_name.split('_')
        year = name_split[0][2:]
        zoom = name_split[1]
        x = name_split[2]
        y = name_split[3][:-4]
        url = URL.format(z=zoom,x=x,y=y)
        forest_urls.append(url)
    return forest_urls

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
def main():

    out_dir = '/mnt/ds3lab-scratch/lming/data/min_quality'
    out_forest_dir = os.path.join(out_dir, 'forest_gain')
    hansen_dir = os.path.join(out_dir, 'hansen')
    create_dir(out_forest_dir)

    # Get hansen files
    hansen_files = getListOfFiles(hansen_dir)
    forest_urls = get_forest_urls(hansen_files, FOREST_GAIN_URL)
    # print(len(forest_urls))
    # print(len(hansen_files))
    # getKeyFromFiles(hansen_files)

    for chunk in chunks(forest_urls, 16):
        with multiprocessing.Pool(processes=16) as pool:
            results = pool.starmap(download_item, create_tuple(chunk, out_forest_dir, 'forest'))

if __name__ == '__main__':
    main()
