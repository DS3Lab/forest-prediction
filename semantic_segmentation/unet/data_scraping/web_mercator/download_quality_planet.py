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

MOSAICS_URL = "https://tiles.planet.com/basemaps/v1/planet-tiles/global_quarterly_{year}{q}_mosaic/gmap/{z}/{x}/{y}.png?api_key=25647f4fc88243e2a6e91150aaa117e3"
MOSAIC_MONTH_URL = 'https://tiles.planet.com/basemaps/v1/planet-tiles/global_monthly_{year}_{month}_mosaic/gmap/{z}/{x}/{y}.png?api_key=25647f4fc88243e2a6e91150aaa117e3',
logger = logging.getLogger('download-planet')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('download-planet.log')
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

def download_item(url, folder, item_type='planet'):
    """
    Download quad tif
    """
    if item_type == 'hansen':
        local_filename = '_'.join(url.split('/')[5:])
        version = local_filename[4:8]
        if version == 'v1.5':
            prefix = 'ly2017_'
        else: # 2018
            prefix = 'ly2018_'
        local_filename = prefix + '_'.join(local_filename.split('_')[-3:])
    elif item_type == 'planet':
        local_filename = '_'.join(url.split('/')[-5:]).split('?')[0]
        name_split = local_filename.split('_')
        year, quarter = name_split[2][:4], name_split[2][4:]
        zoom, x, y = name_split[-3], name_split[-2], name_split[-1]
        local_filename = 'pl' + year + '_' + quarter + '_' + zoom + '_' + x + '_' + y

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

def get_planet_urls(hansen_files, timelapse='quarter'):
    assert timelapse in ['quarter', 'month']
    planet_urls = []
    for file in hansen_files:
        file_name = file.split('/')[-1]
        name_split = file_name.split('_')
        year = name_split[0][2:]
        zoom = name_split[1]
        x = name_split[2]
        y = name_split[3][:-4]
        if timelapse == 'quarter':
            mosaics = [
                MOSAICS_URL.format(year=year, q='q1', z=zoom, x=x, y=y),
                MOSAICS_URL.format(year=year, q='q2', z=zoom, x=x, y=y),
                MOSAICS_URL.format(year=year, q='q3', z=zoom, x=x, y=y),
                MOSAICS_URL.format(year=year, q='q4', z=zoom, x=x, y=y),
                MOSAICS_URL.format(year=int(year)-1, q='q1', z=zoom, x=x, y=y),
                MOSAICS_URL.format(year=int(year)-1, q='q2', z=zoom, x=x, y=y),
                MOSAICS_URL.format(year=int(year)-1, q='q3', z=zoom, x=x, y=y),
                MOSAICS_URL.format(year=int(year)-1, q='q4', z=zoom, x=x, y=y)
            ]
        else: # month
            mosaics = [
                MOSAIC_MONTH_URL.format(year=year, month='01', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year, month='02', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year, month='03', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year, month='04', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year, month='05', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year, month='06', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year, month='07', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year, month='08', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year, month='09', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year, month='10', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year, month='11', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year, month='12', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year-1, month='01', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year-1, month='02', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year-1, month='03', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year-1, month='04', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year-1, month='05', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year-1, month='06', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year-1, month='07', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year-1, month='08', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year-1, month='09', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year-1, month='10', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year-1, month='11', z=zoom, x=x, y=y),
                MOSAIC_MONTH_URL.format(year=year-1, month='12', z=zoom, x=x, y=y)
            ]
        planet_urls.extend(mosaics)
    return planet_urls

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

def main():
    out_dir = '/mnt/ds3lab-scratch/lming/data/min_quality'
    out_planet_dir = os.path.join(out_dir, 'planet_monthly')
    hansen_dir = os.path.join(out_dir, 'forest_loss_yearly')
    create_dir(out_planet_dir)

    # Get hansen files
    hansen_files = getListOfFiles(hansen_dir)
    planet_urls = get_planet_urls(hansen_files, 'month')


    for chunk in chunks(planet_urls, 16):
        with multiprocessing.Pool(processes=16) as pool:
            results = pool.starmap(download_item, create_tuple(chunk, out_planet_dir, 'planet'))

if __name__ == '__main__':
    main()
