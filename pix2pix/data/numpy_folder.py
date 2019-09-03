"""
Does the same as image_folder but for npy files
"""
import numpy as np
import os
import glob
import cv2

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
    return images[:min(max_dataset_size, len(images))]

def make_planet_dataset(dir, target_dir, max_dataset_size=float("inf")):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    images = glob.glob(os.path.join(dir, '*'))
    image_pairs = []
    for img in images:
        image_pairs.append((img, get_target(img, target_dir)))
    return image_pairs[:min(max_dataset_size, len(images))]

def get_target(filename, target_dir):
    items = filename.split('/')[-1].split('_')
    target_name = '_'.join((items[0], items[2], items[3],
        items[4], items[5], items[6][:-4])) + '.npy'
    target_name = os.path.join(target_dir, target_name)
    return target_name

def open_image(img_path):
    # For numpy assume that it has CHW format, and are not between 0 and 1
    if img_path[-4:] == '.npy':
        img_arr = np.load(img_path)
        # return HWC format
        img_arr = img_arr.transpose([1,2,0]) # 64, 64, 3
        return img_arr / 255.
    else: # png
        print('OPEN IMAGE IN NUMPY_THING', img_path)
        img_arr = cv2.imread(img_path)
        return cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
