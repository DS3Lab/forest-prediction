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

def make_planet_dataset(dir, target_dir, max_dataset_size=float("inf")):
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    images = glob.glob(dir)
    image_pairs = []
    for img in images:
        image_pairs.append(img, get_target(img, target_dir))
    return image_pairs[:min(max_dataset_size, len(images))]

def get_target(filename, target_dir):
    items = key.split('/')[-1].split('_')
    target_name = '_'.join((items[0], items[2], items[3],
        items[4], items[5], items[6][:-4])) + '.npy'
    target_name = os.path.join(target_dir, target_name)
    return target_name

def open_numpy(img_path):
    img_arr = np.load(img_path)
    # return HWC format
    if img_path.shape[0] == 3:
        return img_arr.transpose([1,2,0])
    else:
        return img_arr
