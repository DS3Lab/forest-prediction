"""Module to download "high quality" hansen tiles (min 5% forest loss)
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

logger = logging.getLogger('donwload')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('download.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def num2deg(xtile, ytile, zoom):
  n = 2.0 ** zoom
  lon_deg = xtile / n * 360.0 - 180.0
  lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
  lat_deg = math.degrees(lat_rad)
  return (lon_deg, lat_deg)

def deg2num(lon_deg, lat_deg, zoom):
  lat_rad = math.radians(lat_deg)
  n = 2.0 ** zoom
  xtile = int((lon_deg + 180.0) / 360.0 * n)
  ytile = int((1.0 - math.log(math.tan(lat_rad) + (1 / math.cos(lat_rad))) / math.pi) / 2.0 * n)
  return (xtile, ytile)

def bbox2tiles(bbox, zoom):
    """ Return tile coordinates from a bounding box"""
    upper_left_tile = deg2num(bbox['upper_left'][0], bbox['upper_left'][1], zoom)
    lower_right_tile = deg2num(bbox['lower_right'][0], bbox['lower_right'][1], zoom)
    tile_coords = []
    for i in range(upper_left_tile[0], lower_right_tile[0]+1):
        for j in range(upper_left_tile[1], lower_right_tile[1]+1):
            tile_coords.append((i,j))

    return tile_coords

def gen_hansen_urls(tile_coords, zoom):
    urls = []
    for x, y in tile_coords:
        urls.append(HANSEN18_URL.format(z=zoom, x=x, y=y))
        urls.append(HANSEN17_URL.format(z=zoom, x=x, y=y))
    return urls

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n] # yields a list of size n


def create_tuple(list_, name1, name2):
    return [(l, name1, name2) for l in list_]

def download_item(url, folder, item_type):
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
    # create_dir(folder) # This shouldn't be in the method I think
    # print('Downloading', local_filename, 'storing in', os.path.join(folder, local_filename))
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
        
        # Check if it is a quality image
    img_arr = cv2.imread(path)
    if check_quality_label(img_arr):
        logger.debug('QUALITY IMAGE: ' + url)
        return path
    else:
        try:
            os.remove(path)
            logger.debug('REMOVED IMAGE: ' + url)
        except:
            logger.debug('something happened while trying to remove: ' + path)
            return None


def check_quality_label(img, threshold = 0.02):
    count_nonzero = np.count_nonzero(img[:,:,2])  # asume BGR, labels in red channel
    img_size = img[:,:,2].size
    print('nonzeros is',count_nonzero)
    if (count_nonzero / img_size) >= threshold:
        return True
    else:
        return False


def main():
	bbox = {
		'upper_left': (-84.04511825722398, 13.898213869443307),
		'lower_right': (-38.082088, -52.993502)
	}
	zoom = 12
	tile_coords = bbox2tiles(bbox, zoom)
	hansen_urls = gen_hansen_urls(tile_coords, zoom)
	out_dir = '/mnt/ds3lab-scratch/lming/data/min_quality'
	out_hansen_dir = os.path.join(out_dir, 'hansen')
	create_dir(out_dir)
	create_dir(out_hansen_dir)
	
	for chunk in chunks(hansen_urls, 16):
            with multiprocessing.Pool(processes=16) as pool:
                results = pool.starmap(download_item, create_tuple(chunk, out_hansen_dir, 'hansen'))

if __name__ == '__main__':
    main()




