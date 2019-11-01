"""This module produces annual mosaics from quarter mosaics in Planet Data
"""
import cv2
import numpy as np
import glob
import os
from utils import create_dir

PLANETPATH = '/mnt/ds3lab-scratch/lming/gee_data/ldpl/planet' 
# years = ['2016', '2017']
years = ['2016', '2017','2018']
OUTPATH = '/mnt/ds3lab-scratch/lming/gee_data/ldpl/planet/annual'
create_dir(OUTPATH)
for year in years:
    create_dir(os.path.join(OUTPATH, year))

def annual_mosaic(imgs):
    n = 0
    annual = np.zeros(imgs[0].shape)
    for img in imgs:
        annual += img
        n += 1
    return annual / n

def get_imgs(path):
    all_imgs = glob.glob(path)
    imgs = {}
    for img in all_imgs:
        # pl{year}_{q}_{z}_{x}_{y}.png
        items = img.split('/')[-1].split('_')
        year, z, x, y = items[0][2:], items[2], items[3], items[4][:-4]
        key = '_'.join((year, z, x, y))
        imgs[key] = 'pl{year}_{z}_{x}_{y}.npy'.format(year=year, z=z, x=x, y=y)
    return imgs

quarters = ['q1', 'q2', 'q3', 'q4']

for year in years:
    imgs = get_imgs(os.path.join(PLANETPATH, year, '*.png'))
    for key in imgs:
        year, z, x, y = key.split('_')
        if not os.path.exists(os.path.join(OUTPATH, year, imgs[key])):
            qs = [cv2.imread(os.path.join(PLANETPATH, year, 'pl{year}_{q}_{z}_{x}_{y}.png').format(year=year, q=q, z=z, x=x, y=y)) for q in quarters]
            annual = annual_mosaic(qs)
            np.save(os.path.join(OUTPATH, year, imgs[key]), annual)

