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

forest_path2013 = '/mnt/ds3lab-scratch/lming/data/min_quality11/forest_loss/2013'
loss_files = glob.glob(os.path.join(forest_path2013, '*'))

zero2one = []
one2two = []
two2three = []
three2four = []
four2five = []
five2ten = []
tenplus = []

def open_img(path):
    img_arr = cv2.imread(forest_loss_path)
    img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2GRAY)
    return img_arr

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n] # yields a list of size n

def create_tuple(list_):
    return [(l) for l in list_]

def check_quality_label(img_path):
    img = open_img(img_path)
    count_nonzero = np.count_nonzero(img)
    img_size = img.size
    pct = count_nonzero / img_size
    if pct < 0.01:
        zero2one.append(img_path)
    elif 0.01 <= pct < 0.02:
        one2two.append(img_path)
    elif 0.02 <= pct < 0.03:
        two2three.append(img_path)
    elif 0.03 <= pct < 0.04:
        three2four.append(img_path)
    elif 0.04 <= pct < 0.05:
        four2five.append(img_path)
    elif 0.05 <= pct < 0.1:
        five2ten.append(img_path)
    else: # 10 plus
        tenplus.append(img_path)

def main():
	for loss_file_chunk in chunks(loss_files, 16):
            with multiprocessing.Pool(processes=16) as pool:
                results = pool.starmap(check_quality_label, create_tuple(loss_file_chunk))
    stats = {
        'zero2one': zero2one,
        'one2two': one2two,
        'two2three': two2three,
        'three2four': three2four,
        'four2five': four2five,
        'five2ten': five2ten,
        'tenplus': tenplus
    }
    with open('download_stats.pkl', 'wb') as pkl_file:
        pkl.dump(stats, pkl_file)

if __name__ == '__main__':
    main()
