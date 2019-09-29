import os
import glob
import rasterio
import gdal
import numpy as np
import math
import argparse
import matplotlib.pyplot as plt
from utils import deg2num, num2deg, geodesic2spherical, create_dir
from rasterio.merge import merge
from rasterio.windows import Window

def get_tiles(path):
    files = glob.glob(os.path.join(path, '*.png'))
    tiles = []
    for f in files:
        items = f.split('/')[-1].split('_')
        tiles.append('_'.join((items[1], items[2], items[3][-:4])))
    return tiles

def extract_tile(bigtif, zoom, lon, lat, tile_size, crs):
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

    # idx, idy = bigtif.index(x,y)
    # return np.copy(bigtif[idx:idx+256, idy:idy+256])


def main(args):
    path = '/mnt/ds3lab-scratch/lming/data/min_quality12/forest_cover_raw'
    tiles = get_tiles(path)
    vrt_list = ['/mnt/ds3lab-scratch/lming/gee_data/landsat2013.vrt',
                '/mnt/ds3lab-scratch/lming/gee_data/landsat2014.vrt',
                '/mnt/ds3lab-scratch/lming/gee_data/landsat2015.vrt',
                '/mnt/ds3lab-scratch/lming/gee_data/landsat2016.vrt',
                '/mnt/ds3lab-scratch/lming/gee_data/landsat2017.vrt',
                '/mnt/ds3lab-scratch/lming/gee_data/landsat2018.vrt']
    mosaics = [rasterio.open(vrt) for vrt in vrt_list]
    tile_size = 256
    # year = args.input_dir.split('/')[-1] # assume is /mnt/ds3-..../2017
    name_template = 'ld{year}_{z}_{x}_{y}.npy'
    years = [mosaic.split('/')[-1][-8:-4] for mosaic in mosaics]
    for year in years:
        create_dir(os.path.join(args.out_dir, year))
    for tile in tiles:
        zoom, x, y = tile.split('_')
        lon, lat = num2deg(x, y, zoom) # this is ESPG: 4326
        for mosaic in mosaics:
            img_arr = extract_tile(mosaic, lon, lat, tile_size, crs='ESPG:4326')
            year = mosaic.split('/')[-1][-8:-4]
            outname = os.path.join(args.out_dir, year, name_template.format(year=year, zoom=z, x=x, y=y))
            np.save(outname, img_arr)

if __name__ == '__init__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input_dir", type=str, help="directory containing the tifs from gee")
    parser.add_argument("--output_dir", type=str, help="output directory")
    args = parser.parse_args()
    main(args)
