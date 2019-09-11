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

MOSAIC_URL = "https://tiles.planet.com/basemaps/v1/planet-tiles/global_quarterly_{year}{q}_mosaic/gmap/{z}/{x}/{y}.png?api_key=25647f4fc88243e2a6e91150aaa117e3"
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

def zoom2zoom(z0, x0, y0, z1):
    """
    Get the corresponding tiles from z0_x0_y0 with zoom z1
    Right now: z0=12, z1=16
    Return: new upper left x,y of z1, and number of subsequent tiles in one direction
    """
    assert z0 < z1
    zoom_dif = z1 -z0
    x1 = 2**zoom_dif*x0 # corresponding x from the same upper left in zoom z1 coordinate
    y1 = 2**zoom_dif*y0 # corresponding y from the same upper left in zoom z1 coordinate
    num_tiles = 2**zoom_dif
    return x1, y1, num_tiles

def zoom2tiles(z0, x0, y0, z1):
    new_x, new_y, num_tiles = zoom2zoom(z0, x0, y0, z1)
    tiles = []
    for i in range(num_tiles):
        for j in range(num_tiles):
            tiles.append((new_x+i, new_y+j))
    return tiles

def get_planet_urls(hansen_files):
    planet_urls = []
    for file in hansen_files:
        file_name = file.split('/')[-1]
        name_split = file_name.split('_')
        year = name_split[0][2:]
        z = name_split[1]
        x = name_split[2]
        y = name_split[3][:-4]
        tiles = zoom2tiles(int(z), int(x), int(y), 16)
        for tile in tiles:
            tile_x, tile_y = tile
            mosaics = [
                MOSAIC_URL.format(year=year, q='q1', z=16, x=tile_x, y=tile_y),
                MOSAIC_URL.format(year=year, q='q2', z=16, x=tile_x, y=tile_y),
                MOSAIC_URL.format(year=year, q='q3', z=16, x=tile_x, y=tile_y),
                MOSAIC_URL.format(year=year, q='q4', z=16, x=tile_x, y=tile_y)
            ]
        planet_urls.extend(mosaics)
    return planet_urls

def main():

    hansen_dir = '/mnt/ds3lab-scratch/lming/data/min_quality/forest_cover_processed/no_pct/nips'
    out_planet_dir = '/mnt/ds3lab-scratch/lming/data/min_quality/planet/forest_cover_3m_nips'
    create_dir(out_planet_dir)
    hansen_files = glob.glob(os.path.join(hansen_dir, '*'))
    # Get hansen files
    planet_urls = get_planet_urls(hansen_files)
    assert len(planet_urls) == len(hansen_files) * 256

    for chunk in chunks(planet_urls, 16):
        with multiprocessing.Pool(processes=16) as pool:
            results = pool.starmap(download_item, create_tuple(chunk, out_planet_dir, 'planet'))

if __name__ == '__main__':
    main()
