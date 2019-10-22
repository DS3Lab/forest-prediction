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
from gee_tiles2 import extract_tile
from rasterio.merge import merge
from rasterio.windows import Window
from multiprocessing import Process

def extract_forma_tiles(tiles, year, forma_db, forest_loss_dir):
    # TODO: CHANGE SAVE_FC SAVING MODE!!!!!!!
    out_fl = os.path.join(forest_loss_dir, year)
    create_dir(out_fl)
    fl_template = 'fl{year}_{z}_{x}_{y}.npy'
    for z, x, y in tiles:
        out_name = os.path.join(out_fl, fl_template.format(z=z, x=x, y=y))
        lon, lat = num2deg(int(x), int(y), int(z))
        int_year = int(year[2:])
        img_arr = extract_tile(hansen_db, lon, lat, 256, crs='ESPG:4326')
        save_fc(img_arr, out_name, int_year)

def main():
    gee_dir = '/mnt/ds3lab-scratch/lming/gee_data/'
    with open('/mnt/ds3lab-scratch/lming/gee_data/forma_tiles2017.pkl', 'rb') as f:
        tiles = pkl.load(f)

    forma_dir = os.path.join(gee_dir, 'forma')
    forma2017db = rasterio.open('/mnt/ds3lab-scratch/lming/gee_data/forma/forma2017.vrt')
    out_dir = os.path.join(forma_dir, '2017')
    create_dir(out_dir)
    forma_name = 'forma{year}_{z}_{x}_{y}.npy'
    for tile in tiles:
        z, x, y = tile
        lon, lat = num2deg(int(x), int(y), int(z))
        tile = extract_tile(forma2017db, lon, lat, 256, crs='ESPG:4326')[0]
        np.save(os.path.join(out_dir, forma_name.format(year='2017', z=z, x=x, y=y)), tile)

if __name__=='__main__':
    main()
