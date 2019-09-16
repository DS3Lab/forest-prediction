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

logger = logging.getLogger('move_tiles')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('move_tiles.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

def open_img(path):
    img_arr = cv2.imread(path)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    return img_arr

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n] # yields a list of size n

def create_tuple(list_, name1):
    return [(l, name1) for l in list_]

def get_dw_tiles(files):
    # returns x, y
    tiles = []
    for file in files:
        items = file.split('/')[-1].split('_')
        tiles.append((items[1], items[2], items[3][:-4]))
    return tiles

def mv_forest_cover(forest_cover_path, tile):
    forest_cover_files = glob.glob(os.path.join(forest_cover_path, '*'))
    create_dir()

def mv_tiles(tile, fc_path, fg_path, fl_path, landsat_path, planet_path):

    z, x, y = tile[0], tile[1], tile[2]
    years = ['2013', '2014', '2015', '2016', '2017']
    quarters = ['q1', 'q2', 'q3', 'q4']
    forest_cover_name = 'fc2000_{z}_{x}_{y}.png'.format(z=z, x=x, y=y)
    forest_gain_name = 'fg2012_{z}_{x}_{y}.png'.format(z=z, x=x, y=y)
    forest_loss_name = 'fl{year}_{z}_{x}_{y}.png'
    landsat_name = 'ld{year}_{z}_{x}_{y}.png'
    planet_name = 'pl{year}_{q}_{z}_{x}_{y}.png'
    # planet_names = [planet_name.format(q=q, z=z, x=x, y=y) for q in quarters]

    # Move forest cover
    if os.path.exists(os.path.join(fc_path, forest_cover_name)):
        os.rename(os.path.join(fc_path, forest_cover_name),
            os.path.join(fc_path, 'min_pct', forest_cover_name))
    else:
        logger.debug('Not exist: ' + forest_cover_name)
    # Move forest gain
    if os.path.exists(os.path.join(fg_path, forest_gain_name)):
        os.rename(os.path.join(fg_path, forest_gain_name),
            os.path.join(fg_path, 'min_pct', forest_gain_name))
    else:
        logger.debug('Not exist: ' + forest_gain_name)

    for year in years:
        # Move forest loss
        forest_loss = forest_loss_name.format(year=year, z=z, x=x, y=y)
        if os.path.exists(os.path.join(fl_path, year, forest_loss)):
            os.rename(os.path.join(fl_path, year, forest_loss),
                os.path.join(fl_path, 'min_pct', year, forest_loss))
        else:
            logger.debug('Not exist: ' + forest_loss)
        # Move landsat
        landsat = landsat_name.format(year=year, z=z, x=x, y=y)
        if os.path.exists(os.path.join(landsat_path, year, landsat)):
            os.rename(os.path.join(landsat_path, year, landsat),
                os.path.join(landsat_path, 'min_pct', year, landsat))
        else:
            logger.debug('Not exist: ' + landsat)
        # Move planet
        for quarter in quarters:
            planet = planet_name.format(year=year, q=quarter, z=z, x=x, y=y)
            if os.path.exists(os.path.join(planet_path, year, planet)):
                os.rename(os.path.join(planet_path, year, planet),
                    os.path.join(planet_path, 'min_pct', year, planet))
            else:
                logger.debug('Not exist: ' + planet)
def main():
    # for chunk in chunks(loss_files[:100], 10):
    #     with multiprocessing.Pool(processes=16) as pool:
    #         results = pool.starmap(check_quality_label, create_tuple(chunk, 'hello'))
    with open('download_stats.pkl', 'rb') as pkl_file:
        stats = pkl.load(pkl_file)

    files = stats['one2two'] + stats['two2three'] + stats['three2four'] + \
        stats['four2five'] + stats['five2ten'] + stats['tenplus']
    tiles = get_dw_tiles(files)

    fc_path = '/mnt/ds3lab-scratch/lming/data/min_quality11/forest_cover'
    fg_path = '/mnt/ds3lab-scratch/lming/data/min_quality11/forest_gain'
    fl_path = '/mnt/ds3lab-scratch/lming/data/min_quality11/forest_loss'
    landsat_path = '/mnt/ds3lab-scratch/lming/data/min_quality11/landsat'
    planet_path = '/mnt/ds3lab-scratch/lming/data/min_quality11/planet'
    years = ['2013', '2014', '2015', '2016', '2017']
    create_dir(os.path.join(fc_path, 'min_pct'))
    create_dir(os.path.join(fg_path, 'min_pct'))
    create_dir(os.path.join(fl_path, 'min_pct'))
    create_dir(os.path.join(landsat_path, 'min_pct'))
    create_dir(os.path.join(planet_path, 'min_pct'))
    for year in years:
        create_dir(os.path.join(fl_path, 'min_pct', year))
        create_dir(os.path.join(landsat_path, 'min_pct', year))
        create_dir(os.path.join(planet_path, 'min_pct', year))

    for tile in tiles:
        mv_tiles(tile, )


if __name__ == '__main__':
    main()
