"""Module to download "high quality" hansen tiles (min 5% forest loss)
WARNING: DOESN'T WORK, REMOVE
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


logger = logging.getLogger('log')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('log.log')
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

def download_file(url, path):
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
        return 1
    except:
        logger.debug('FAILED: ' + url)
        return None

def get_tile_from_url(url):
    items = url.split('/')
    z, x, y = items[-3], items[-2], items[-1][:-4]
    return z,x,y

def download_item(url, folder, item_type):
    """
    Download quad tif
    """
    assert item_type == 'hansen'
    z, x, y = get_tile_from_url(url)

    local_filename = '_'.join(url.split('/')[5:])
    version = local_filename[4:8]
    if version == 'v1.5':
        prefix = 'ly2017_'
        prefix2 = 'ly2016_'
        url2 = HANSEN16_URL.format(z=z,x=x,y=y)
        local_filename2 = '_'.join(url2.split('/')[5:])
    else: # 2018
        prefix = 'ly2018_'
        prefix2 = 'ly2017_'
        url2 = HANSEN17_URL.format(z=z,x=x,y=y)
        local_filename2 = '_'.join(url2.split('/')[5:])
    local_filename = prefix + '_'.join(local_filename.split('_')[-3:])
    local_filename2 = prefix2 + '_'.join(local_filename2.split('_')[-3:])
    if folder:
        path = os.path.join(folder, local_filename)
        path2 = os.path.join(folder, local_filename2)
    else:
        path = local_filename
        path2 = local_filename2
    # if os.path.isfile(os.path.join(folder, local_filename)):
    #     return os.path.join(folder, local_filename)
    df1 = download_file(url, path)
    df2 = download_file(url2, path2)
    if df1 == 1 and df2 == 1: # both good downloading
        logger.debug('DOWNLOADED: ' + url + ' ' + url2)
            # Check if it is a quality image
        img_arr = cv2.imread(path)
        img_arr2 = cv2.imread(path2)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
        img_arr2 = cv2.cvtColor(img_arr2, cv2.COLOR_BGR2GRAY)
        img = img_arr - img_arr2
        if check_quality_label(img):
            logger.debug('QUALITY IMAGE: ' + url)
            np.save(prefix + '_'.join(z,x,y) + '.npy', img.astype(np.uint8))
            return prefix + '_'.join(z,x,y) + '.npy'
        else:
            try:
                os.remove(path)
                logger.debug('REMOVED IMAGE: ' + url)
            except:
                logger.debug('something happened while trying to remove: ' + path)
                return None
            try:
                os.remove(path2)
                logger.debug('REMOVED IMAGE: ' + url2)
            except:
                logger.debug('something happened while trying to remove: ' + path2)
                return None


def check_quality_label(img, threshold = 0.005):
    """
    img = np.array(256,256)
    """
    count_nonzero = np.count_nonzero(img)  # asume BGR, labels in red channel
    img_size = img.size
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
	out_dir = '/mnt/ds3lab-scratch/lming/data/min_quality/final'
	out_hansen_dir = os.path.join(out_dir, 'hansen')
	create_dir(out_dir)
	create_dir(out_hansen_dir)

	for chunk in chunks(hansen_urls, 16):
            with multiprocessing.Pool(processes=16) as pool:
                results = pool.starmap(download_item, create_tuple(chunk, out_hansen_dir, 'hansen'))

if __name__ == '__main__':
    main()
