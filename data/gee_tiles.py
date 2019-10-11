"""This module extracts the corresponding tiles of web mercator from GEE"""
import os
import glob
import rasterio
import gdal
import numpy as np
import math
import argparse
import matplotlib.pyplot as plt
import pickle as pkl
import logging
from utils import deg2num, num2deg, geodesic2spherical, create_dir
from rasterio.merge import merge
from rasterio.windows import Window
from multiprocessing import Process
# from time import sleep

logger = logging.getLogger('gee')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('gee.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

def extract_tile(mosaicdb, lon, lat, tile_size, crs):
    '''
    Extract tile_size x tile_size tile from a bif tif starting from lon, lat
    bigtif: Rasterio Data Reader
    '''
    assert crs in ['ESPG:3857', 'ESPG:4326']
    if crs == 'ESPG:4326':
        xgeo, ygeo = geodesic2spherical(lon, lat)
    else:
        xgeo, ygeo = lon, lat
    idx, idy = mosaicdb.index(xgeo, ygeo)
    return mosaicdb.read(window=Window(idy, idx, 256, 256))

def preprocess_fc(img_arr, threshold=0.25):
    """Threshold percentage values to be binary to 0.25
    """
    if img_arr.max() == 100.:
        img_arr = img_arr / 100.
    fc_mask = np.where(img_arr >= threshold)
    nfc_mask = np.where(img_arr < threshold)
    img_arr[fc_mask] = 1
    img_arr[nfc_mask] = 0
    return img_arr

def create_forest_cover(fc2000, gain2000_2012, loss2000_2012, loss2013_year):
    """
    Params:
    fc2000: ndarray binary mask
    gain2000_2012: ndarray binary mask
    loss2000_2012: ndarray binary mask
    loss2013_year: ndarray binary mask
    """
    gain_loss2000_2012 = gain2000_2012 - loss2000_2012 # 0 if loss and gain, 1 if only gain, -1 if only loss
    gain_mask = np.where(gain_loss2000_2012==1)
    loss_mask = np.where(gain_loss2000_2012==-1)
    loss2013_year_mask = np.where(loss2013_year==1)
    # Update forest cover
    fc2000[gain_mask] = 1
    fc2000[loss_mask] = 0
    fc2000[loss2013_year_mask] = 0
    return fc2000

def get_aggregated_loss(img_arr, beg=1, end=12):
    """Gets the loss from 2001 to 2012
    """
    loss_arr = np.zeros(img_arr.shape)
    for i in range(beg, end+1): # +1 because range is exclusive
        mask = np.where(img_arr==i)
        loss_arr[mask] = 1
    return loss_arr

def gen_tile(img_db, lon, lat, tile_type, year, out_name):
    """
    year: int, from 1 to 18
    """
    # 'treecover2000', 'gain', 'lossyear', 'lossyear2000_2012')
    assert tile_type in ['ld', 'fc', 'fl']
    FC_IDX = 0 # forest cover index
    GAIN_IDX = 1 # forest gain index
    LOSS_IDX = 2 # forest loss index
    if tile_type == 'fc':
        loss_arr = extract_tile(img_db[LOSS_IDX], lon, lat, 256, crs='ESPG:4326')
        loss2000_2012 = get_aggregated_loss(loss_arr, beg=1, end=12) # 0 is no loss in this band, get loss from [1,12]
        gain2000_2012 = extract_tile(img_db[GAIN_IDX])
        loss2013_year = get_aggregated_loss(loss_arr, beg=13, end=year)
        fc2000 = extract_tile(img_db[FC_IDX], lon, lat, 256, crs='ESPG:4326')
        img_arr = create_forest_cover(fc2000, gain2000_2012, loss2000_2012, loss2013_year)
    elif tile_type == 'fl':
        img_arr = extract_tile(img_db[LOSS_IDX], lon, lat, 256, crs='ESPG:4326')
        loss_mask = np.where(img_arr == year)
        no_loss_mask = np.where(img_arr != year)
        img_arr[loss_mask] = 1
        img_arr[no_loss_mask] = 0
    else:
        img_arr = extract_tile(img_db, lon, lat, 256, crs='ESPG:4326')
    np.save(out_name, img_arr)


def extract_tiles(tiles, year, landsat_db, hansen_db, landsat_dir, forest_cover_dir, forest_loss_dir):
    out_landsat = os.path.join(landsat_dir, year)
    out_fc = os.path.join(forest_cover_dir, year)
    out_fl = os.path.join(forest_loss_dir, year)
    create_dir(out_landsat)
    create_dir(out_fc)
    create_dir(out_fl)
    landsat_template = 'ld{year}_{z}_{x}_{y}.npy'
    fc_template = 'fc{year}_{z}_{x}_{y}.npy'
    fl_template = 'fl{year}_{z}_{x}_{y}.npy'
    for z,x,y in [(12, 1260, 2185)]:
    # for z, x, y in tiles:
        lon, lat = num2deg(int(x), int(y), int(z))
        int_year = int(year[2:])
        gen_tile(landsat_db, lon, lat, 'ld', year, os.path.join(out_landsat, landsat_template.format(year=year, z=z, x=x, y=y)))
        gen_tile(hansen_db, lon, lat, 'fc', year, os.path.join(out_fc, fc_template.format(year=year, z=z, x=x, y=y)))
        gen_tile(hansen_db, lon, lat, 'fl', year, os.path.join(out_fl, fl_template.format(year=year, z=z, x=x, y=y)))

def main():
    gee_dir = '/mnt/ds3lab-scratch/lming/gee_data'
    landsat_dir = os.path.join(gee_dir, 'ls7')

    with open('tiles.pkl', 'rb') as f:
        tiles = pkl.load(f)

    forest_cover_dir = os.path.join(gee_dir, 'forest_cover')
    forest_loss_dir = os.path.join(gee_dir, 'forest_loss')
    create_dir(forest_cover_dir)
    create_dir(forest_loss_dir)
    years = ['2013', '2014', '2015', '2016', '2017', '2018']
    landsat_dbs = {}
    for year in years:
        create_dir(os.path.join(forest_cover_dir, year))
        create_dir(os.path.join(forest_loss_dir, year))
        lansdat_dbs[year] = rasterio.open(os.path.join('ls7', 'landsat' + year + '.vrt'))
    hansen_db = rasterio.open(os.path.join(gee_dir, 'hansen.vrt'))

    processes = []
    for year in years:
        p = Process(target=extract_tiles, args=(tiles, year, landsat_dbs[year], hansen_db, landsat_dir,
                forest_cover_dir, forest_loss_dir,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

if __name__ == '__main__':
    main()
