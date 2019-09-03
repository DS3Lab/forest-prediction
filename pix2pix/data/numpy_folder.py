"""
Does the same as image_folder but for npy files
"""
import numpy as np
import os
import glob

def make_dataset(dir, keyname, years, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    if years == 'all':
        years = list(next(os.walk(dir))[1])
    else:
        years = years.split(',')
    # for subdir in next(os.walk(dir))[1]:
    for year in years:
        images.extend([file for file in glob.glob(os.path.join(dir, year, keyname + '*'))])
    print('MAKE DATASET', images, os.path.join(dir, keyname + '*'))
    return images[:min(max_dataset_size, len(images))] 

