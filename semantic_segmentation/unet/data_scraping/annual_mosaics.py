import numpy as np
import requests
import rasterio
import os
import glob
import logging
import cv2
from shutil import copyfile
from tqdm import tqdm

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def loadFiles(mask_dir, img_dir, qualities, years, img_type, limit=float('inf')):
    """
    Search for the img masks and stores them in a list

    :param input_dir: Directory where the images are stored.
    :param type: 'quarter' or 'year'
        Dataset format:
        Project/
        |-- input_dir/
        |   |-- hansen/
        |   |   |-- five_pct/
        |   |   |   |-- 2018/
        |   |   |   |   |-- *.png
        |   |   |   |-- 2017/
        |   |   |   |-- ...
        |   |   |-- four_pct/
        |   |   |   |-- 2018/
        |   |   |   |-- 2017/
        |   |   |   |-- ...
        |   |-- planet/
        |   |   |   |-- *.png
    :param years: Years of tiles to be loaded
    """

    imgs = {}
    for quality in qualities:
        for year in years:
            masks_path = os.path.join(mask_dir, quality, year)
            mask_imgs = glob.glob(os.path.join(masks_path, '*.png'))
            for mask in mask_imgs:
                mask_dict = get_quarter_imgs_from_mask(mask, img_dir) \
                    if img_type == 'quarter' \
                    else get_annual_imgs_from_mask(mask, img_dir)
                imgs = {**imgs, **mask_dict}
            if len(imgs) > limit: # soft limit, returns when it is greater
                return imgs
    return imgs

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

def get_quarter_imgs_from_mask(mask_file, img_dir):
    """
    Input img from year t - 1 quarter 1 to quarter 2, loss from year t.
    """
    year, z, x, y = get_tile_info(mask_file.split('/')[-1])
    key = str(year) + '_' + z + '_' + x + '_' + y
    planet_name = 'pl' + '{year}' + '_{q}_{z}_{x}_{y}.png'
    planet_template = os.path.join(img_dir, planet_name)
    data = {}
    data[key] = {
        'img': (planet_template.format(year=year-1, q='q1', z=z, x=x, y=y),
                planet_template.format(year=year-1, q='q2', z=z, x=x, y=y),
                planet_template.format(year=year-1, q='q3', z=z, x=x, y=y),
                planet_template.format(year=year-1, q='q4', z=z, x=x, y=y),
                planet_template.format(year=year, q='q1', z=z, x=x, y=y),
                planet_template.format(year=year, q='q2', z=z, x=x, y=y),
                planet_template.format(year=year, q='q3', z=z, x=x, y=y),
                planet_template.format(year=year, q='q4', z=z, x=x, y=y)),
        'mask': mask_file
    }
    return data

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

def main():
    logger = logging.getLogger('annual-mosaic')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler('annual-mosaic.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    input_dir = '/mnt/ds3lab-scratch/lming/data/min_quality'
    mask_dir = os.path.join(input_dir, 'hansen')
    img_dir = os.path.join(input_dir, 'planet')
    out_planet_dir = os.path.join(img_dir, 'annual')
    create_dir(out_planet_dir)
    qualities = ['five_pct', 'four_pct', 'three_pct', 'two_pct']
    years = ['2017', '2018']
    img_type = 'quarter'

    paths_dict = loadFiles(mask_dir, img_dir, qualities, years, img_type)
    keys = list(paths_dict.keys())
    for i in range(len(keys)):
        key = keys[i]
        path_dict = paths_dict[key]
        images = path_dict['img']
        mask = path_dict['mask']
        key_info = key.split('_')
        year1, z, x, y = key_info[0], key_info[1], key_info[2], key_info[3]
        year0 = str(int(year1) - 1)

        name0 = 'pl' + year0 + '_' + z + '_' + x + '_' + y + '.npy'
        name1 = 'pl' + year1 + '_' + z + '_' + x + '_' + y + '.npy'
        name0 = os.path.join(out_planet_dir, name0)
        name1 = os.path.join(out_planet_dir, name1)

        quads0 = [images[0], images[1], images[2], images[3]]
        quads1 = [images[4], images[5], images[6], images[7]]

        if not os.path.isfile(name0):
            try:
                mosaic0 = gen_yearly_mosaic(quads0)
                np.save(name0, mosaic0)
                logger.debug('SAVED: ' + name0)
            except:
                logger.debug('FAILED: ' + name0)
        if not os.path.isfile(name1):
            try:
                mosaic1 = gen_yearly_mosaic(quads1)
                np.save(name1, mosaic1)
                logger.debug('mosaic saved: ' + name1)
            except:
                logger.debug('FAILED: ' + name1)

if __name__ == '__main__':
    main()