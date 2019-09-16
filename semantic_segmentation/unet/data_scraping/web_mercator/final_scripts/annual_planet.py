import numpy as np
import requests
import rasterio
import os
import glob
import logging
import cv2
import itertools
from shutil import copyfile
from tqdm import tqdm
from crop_tiles import get_hansen_quality_files, get_planet_files

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def get_tile_info(tile):
    """
    Retrieve the year, zoom, x, y from a tile. Example: ly2017_12_1223_2516.png
    """
    tile_items = tile.split('_')
    year = tile_items[0][2:]
    z = tile_items[1]
    x = tile_items[2]
    y = tile_items[3][:-4]
    return int(year), z, x, y

def gen_yearly_mosaic(quads):
    """
    :params: quads: list of 3-month planet mosaic
    Obtained by the pixel-wise median of the mosaics
    """
    src1 = rasterio.open(quads[0])
    src2 = rasterio.open(quads[1])
    src3 = rasterio.open(quads[2])
    src4 = rasterio.open(quads[3])

    r1 = src1.read(1)
    r2 = src2.read(1)
    r3 = src3.read(1)
    r4 = src4.read(1)
    r5 = np.dstack((r1,r2,r3,r4))

    g1 = src1.read(2)
    g2 = src2.read(2)
    g3 = src3.read(2)
    g4 = src4.read(2)
    g5 = np.dstack((g1,g2,g3,g4))

    b1 = src1.read(3)
    b2 = src2.read(3)
    b3 = src3.read(3)
    b4 = src4.read(3)
    b5 = np.dstack((b1,b2,b3,b4))

    r = np.median(r5, axis=2)
    g = np.median(g5, axis=2)
    b = np.median(b5, axis=2)

    yearly_mosaic = np.dstack((r,g,b))
    return yearly_mosaic.transpose([2,0,1]) # NCHW format

def reduce_planet_files(planet_files):
    imgs = {}
    for file in planet_files:
        items = file.split('/')[-1].split('_')
        key = '_'.join((items[0][2:], items[2], items[3], items[4][:-4]))
        if key not in imgs:
            imgs[key] = {
                'year': items[0][2:],
                'z': items[2],
                'x': items[3],
                'y': items[4][:-4]
            }
    return imgs

def get_planet_files(planet_path):
    years = ['2016', '2017']
    planet_files = []
    for year in years:
        planet_files.extend(glob.glob(os.path.join(planet_path, year, '*')))
    return planet_files

def main():
    logger = logging.getLogger('annual_planet')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('annual_planet.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    input_dir = '/mnt/ds3lab-scratch/lming/data/min_quality11/planet/min_pct'
    out_dir = os.path.join(input_dir, 'annual')
    create_dir(out_dir)
    years = ['2016', '2017']
    for year in years:
        create_dir(os.path.join(out_dir, year))
    planet_files = get_planet_files(hansen_files, src_quarter_path)
    planet_files = reduce_planet_files(planet_files)

    for key in planet_files.keys():
        name_template = os.path.join(input_dir, str(year), 'pl{year}_{q}_{z}_{x}_{y}.png')
        img_info = planet_files[key]
        year, z, x, y = img_info['year'], img_info['z'], img_info['x'], img_info['y']
        quads = [
            name_template.format(year=year, q='q1', z=z, x=x, y=y),
            name_template.format(year=year, q='q2', z=z, x=x, y=y),
            name_template.format(year=year, q='q3', z=z, x=x, y=y),
            name_template.format(year=year, q='q4', z=z, x=x, y=y)
        ]
        out_name = os.path.join(out_dir, str(year), 'pl{year}_{z}_{x}_{y}.npy')
        if not os.path.isfile(out_name):
            try:
                mosaic = gen_yearly_mosaic(quads)
                np.save(name, mosaic)
                logger.debug('SAVED: ' + name)
            except:
                logger.debug('FAILED: ' + name)

if __name__ == '__main__':
    main()
