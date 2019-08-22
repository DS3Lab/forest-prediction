""" Script to download quads/mosaics from Planet Labs """

# Plan execution:
# 1.- Get granules from https://earthenginepartners.appspot.com/science-2013-global-forest/download_v1.6.html, which contains the labels.
# 2.- Get the planet mosaics that overlaps with that granule. Right now download the quads from https://api.planet.com/basemaps/v1/mosaics/
# 3.- Merge the mosaics with gdal.
# 4.- Downsample the mosaics to be 30m/px, so it overlaps with Hansen.
# 5.- Merge both planet and Hansen data.
# 6.- Sample where there has been a couple of losses.

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
from itertools import product


HANSEN18_URL = 'https://storage.googleapis.com/earthenginepartners-hansen/tiles/gfc_v1.6/loss_alpha/{z}/{x}/{y}.png'
HANSEN17_URL = 'https://storage.googleapis.com/earthenginepartners-hansen/tiles/gfc_v1.5/loss_alpha/{z}/{x}/{y}.png'
HANSEN16_URL = 'https://storage.googleapis.com/earthenginepartners-hansen/tiles/gfc_v1.4/loss_alpha/{z}/{x}/{y}.png'
MOSAICS_URL = "https://tiles.planet.com/basemaps/v1/planet-tiles/global_quarterly_{year}{q}_mosaic/gmap/{z}/{x}/{y}.png?api_key=25647f4fc88243e2a6e91150aaa117e3"
# MOSAICS_URL = "https://tiles.planet.com/basemaps/v1/planet-tiles/global_quarterly_2017q4_mosaic/gmap/{z}/{x}/{y}.png?api_key=25647f4fc88243e2a6e91150aaa117e3"

logger = logging.getLogger('donwload')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('download.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

# Helper function to printformatted JSON using the json module
def p(data):
    print(json.dumps(data, indent=2))


def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def download_item(url, folder, item_type):
    """
    Download quad tif
    """
    if item_type == 'hansen':
        local_filename = '_'.join(url.split('/')[5:])
    elif item_type == 'planet':
        local_filename = '_'.join(url.split('/')[-5:]).split('?')[0]
    # create_dir(folder) # This shouldn't be in the method I think
    print('Downloading', local_filename, 'storing in', os.path.join(folder, local_filename))
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
        logger.debug('SUCCESS: ' + url)
        return path
    except:
        logger.debug('FAILED: ' + url)
        return None


def num2deg(xtile, ytile, zoom):
  n = 2.0 ** zoom
  lon_deg = xtile / n * 360.0 - 180.0
  lat_rad = math.atan(math.sinh(math.pi * (1 - 2 * ytile / n)))
  lat_deg = math.degrees(lat_rad)
  return (lon_deg, lat_deg)
  # return (lat_deg, lon_deg)


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


def gen_planet_yearly(year, zoom, x, y):
    return [
        MOSAICS_URL.format(year=year, q='q1', z=zoom, x=x, y=y),
        MOSAICS_URL.format(year=year, q='q2', z=zoom, x=x, y=y),
        MOSAICS_URL.format(year=year, q='q3', z=zoom, x=x, y=y),
        MOSAICS_URL.format(year=year, q='q4', z=zoom, x=x, y=y)
    ]

def gen_planet_urls(tile_coords, zoom):
    urls = []
    for x, y in tile_coords:
        urls.extend(gen_planet_yearly(2018, zoom, x, y))
        urls.extend(gen_planet_yearly(2017, zoom, x, y))
        urls.extend(gen_planet_yearly(2016, zoom, x, y))
    return urls


def gen_hansen_urls(tile_coords, zoom):
    urls = []
    for x, y in tile_coords:
        urls.append(HANSEN18_URL.format(z=zoom, x=x, y=y))
        urls.append(HANSEN17_URL.format(z=zoom, x=x, y=y))
        urls.append(HANSEN16_URL.format(z=zoom, x=x, y=y))
    return urls

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n] # yields a list of size n


def create_tuple(list_, name1, name2):
    return [(l, name1, name2) for l in list_]


def main():
    # bbox ={
    #     'upper_left': (-62.27100005274531,-20.286106452852803),
    #     'lower_right': (-58.18408599024531,-24.819844651483027)
    # }
    bbox = {
          'upper_left': (-64.78899751590507,-4.172209480404924),
          'lower_right': (-42.069270953405066,-19.560823763919885)
    }
    zoom = 12
    import pickle as pkl
    with open('re-download.pkl', 'rb') as pkl_file:
        repeat_urls = pkl.load(pkl_file)
    planet_urls = []
    hansen_urls = []
    for url in repeat_urls:
        if 'googleapis' in url:
            hansen_urls.append(url)
        elif 'planet' in url:
            planet_urls.append(url)
   #  tile_coords = bbox2tiles(bbox, zoom)
   #  planet_urls = gen_planet_urls(tile_coords, zoom)
   #  hansen_urls = gen_hansen_urls(tile_coords, zoom)
   #  create_dir('tiles_brazil')
   #  create_dir('tiles_brazil/hansen')
   #  create_dir('tiles_brazil/planet')
    for chunk in chunks(planet_urls, 16):
        with multiprocessing.Pool(processes=16) as pool:
            results = pool.starmap(download_item, create_tuple(chunk, 'tiles_brazil/planet', 'planet'))

    for chunk in chunks(hansen_urls, 16):
        with multiprocessing.Pool(processes=16) as pool:
            results = pool.starmap(download_item, create_tuple(chunk, 'tiles_brazil/hansen', 'hansen'))
    # for url in planet_urls:
    #     download_item(url, 'tiles', 'planet')
if __name__=='__main__':
    main()

