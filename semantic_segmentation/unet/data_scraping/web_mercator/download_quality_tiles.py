"""Module to download "high quality" hansen tiles (min 5% forest loss)
"""
import argparse
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

FOREST_LOSS_URLS = {
    # '2018': 'https://storage.googleapis.com/earthenginepartners-hansen/tiles/gfc_v1.6/loss_alpha/{z}/{x}/{y}.png',
    '2017': 'https://storage.googleapis.com/earthenginepartners-hansen/tiles/gfc_v1.5/loss_alpha/{z}/{x}/{y}.png',
    '2016': 'https://storage.googleapis.com/earthenginepartners-hansen/tiles/gfc_v1.4/loss_alpha/{z}/{x}/{y}.png',
    '2015': 'https://storage.googleapis.com/earthenginepartners-hansen/tiles/gfc_v1.3/loss_alpha/{z}/{x}/{y}.png',
    '2014': 'https://earthengine.google.org/static/hansen_2014/loss_alpha/{z}/{x}/{y}.png',
    '2013': 'https://earthengine.google.org/static/hansen_2013/loss_alpha/{z}/{x}/{y}.png'
}

LANDSAT_URLS = {
    '2017': 'https://storage.googleapis.com/landsat-cache/2017/{z}/{x}/{y}.png,
    '2016': 'https://storage.googleapis.com/landsat-cache/2016/{z}/{x}/{y}.png,
    '2015': 'https://storage.googleapis.com/landsat-cache/2015/{z}/{x}/{y}.png,
    '2014': 'https://storage.googleapis.com/landsat-cache/2014/{z}/{x}/{y}.png,
    '2013': 'https://storage.googleapis.com/landsat-cache/2013/{z}/{x}/{y}.png
}

FOREST_COVER_URL = "https://storage.googleapis.com/earthenginepartners-hansen/tiles/gfc_v1.4/tree_gray/{z}/{x}/{y}.png"

FOREST_GAIN_URL = "https://storage.googleapis.com/earthenginepartners-hansen/tiles/gfc_v1.4/gain_alpha/{z}/{x}/{y}.png"

PLANET_URL = "https://tiles.planet.com/basemaps/v1/planet-tiles/global_quarterly_{year}{q}_mosaic/gmap/{z}/{x}/{y}.png?api_key=25647f4fc88243e2a6e91150aaa117e3"

REDOWNLOAD = []
logger = logging.getLogger('donwload_tiles')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('download_tiles.log')
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

def check_quality_label(img, threshold = 0.02):
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

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n] # yields a list of size n

def create_tuple(list_, name1, name2):
    return [(l, name1, name2) for l in list_]

def download_file(url, path, redo=True):
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
        return 1
    except:
        logger.debug('FAILED:' + url)
        if redo:
            REDOWNLOAD.append((path, url))
        return None

def download_item(tile, z, out_dir):
    """
    Download quad tif
    """
    forest_loss_name = 'fl{year}_{z}_{x}_{y}.png'
    forest_cover_name = 'fc2000_{z}_{x}_{y}.png'
    forest_gain_name = 'fg2012_{z}_{x}_{y}.png'
    planet_name = 'pl{year}_{q}_{z}_{x}_{y}.png'
    landsat_name = 'ld{year}_{z}_{x}_{y}.png'

    years = [2013, 2014, 2015, 2016, 2017]
    quarters = ['q1', 'q2', 'q3', 'q4']

    for year in years:
        forest_loss_path = os.path.join(out_dir, 'forest_loss', str(year), forest_loss_name.format(year=year, z=z, x=tile[0], y=tile[1]))
        forest_loss_url = FOREST_LOSS_URLS[str(year)].format(z=z, x=x, y=y)
        response = download_file(forest_loss_url, forest_loss_path, redo=False)
        # Properly downloaded
        if response == 1:
            if year == 2013:
                img_arr = cv2.imread(forest_loss_path)
                img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
                quality_label = check_quality_label(img)
                if not check_quality_label:
                    break

            forest_cover_path = os.path.join(out_dir, 'forest_cover', forest_cover_name.format(z=z, x=tile[0], y=tile[1]))
            forest_gain_path = os.path.join(out_dir, 'forest_gain', forest_gain_name.format(z=z, x=tile[0], y=tile[1]))
            landsat_path = os.path.join(out_dir, 'landsat', str(year), landsat_name.format(year=year, z=z, x=tile[0], y=tile[1]))

            forest_cover_url = FOREST_COVER_URL.format(z=z, x=tile[0], y=tile[1])
            forest_gain_url = FOREST_GAIN_URL.format(z=z, x=tile[0], y=tile[1])
            landsat_url = LANDSAT_URLS[str(year)].format(z=z, x=tile[0], y=tile[1])
            download_file(forest_cover_url, forest_cover_path)
            download_file(forest_gain_url, forest_gain_path)
            download_file(lansdat_url, landsat_path)
            for q in quarters:
                if year in [2016, 2017, 2018, 2019]: # there are only this years available in planet satellites
                    planet_path = os.path.join(out_dir, 'planet', str(year), planet_name.format(q=q, z=z, x=tile[0], y=tile[1]))
                    planet_url = PLANET_URL.format(year=year, q=q, z=z, x=tile[0], y=tile[1])
                    download_file(planet_url, planet_path)

def check_quality_label(img, threshold = 0.02):
    count_nonzero = np.count_nonzero(img)
    img_size = img.size
    if (count_nonzero / img_size) >= threshold:
        return True
    else:
        return False

def main():
	bbox = {
		'upper_left': (-84.04511825722398, 13.898213869443307),
		'lower_right': (-38.082088, -52.993502)
	}
	zoom = 11
	tile_coords = bbox2tiles(bbox, zoom)
	hansen_urls = gen_hansen_urls(tile_coords, zoom)
	out_dir = '/mnt/ds3lab-scratch/lming/data/min_quality11'
	out_hansen_dir = os.path.join(out_dir, 'hansen')
	create_dir(out_dir)
	create_dir(out_hansen_dir)

	for chunk in chunks(tiles_coords, 16):
            with multiprocessing.Pool(processes=16) as pool:
                results = pool.starmap(download_item, create_tuple(chunk, z, out_hansen_dir))

if __name__ == '__main__':
    main()
