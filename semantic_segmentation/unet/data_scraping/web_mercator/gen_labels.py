import os
import glob
import numpy as np
import rasterio
import logging


logger = logging.getLogger('labels')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('gen-labels.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def get_imgs(path, year):
    path1 = os.path.join(path, str(year))
    path2 = os.path.join(path, str(year-1))
    
    imgs1 = {}
    imgs2 = {}
    
    for file in glob.glob(path1 + '/*.png'):
        key = '_'.join(file.split('/')[-1][:-4].split('_')[-3:])
        imgs1[key] = file
    
    for file in glob.glob(path2 + '/*.png'):
        key = '_'.join(file.split('/')[-1][:-4].split('_')[-3:])
        imgs2[key] = file
    
#     imgs1 = [file for file in glob.glob(path1 + '/*.png')]
#     imgs2 = [file for file in glob.glob(path2 + '/*.png')]
#     assert len(imgs1) == len(imgs2)
    print('LENGHT', len(imgs1))    
    return imgs1, imgs2


def save_labels(imgs1, imgs2, year, out_path):
    for key in imgs1.keys():
        if key not in imgs2:
            logger.debug(key + 'not in ' + str(year - 1))
            continue
        logger.debug('Processing img {}'.format(key))
        img1 = rasterio.open(imgs1[key])
        img2 = rasterio.open(imgs2[key])
        # read only band R
        bands1 = img1.read(1)
        bands2 = img2.read(1)
        ly1 = bands1 - bands2
        ly1 = ly1 / 255.
        np.save(os.path.join(out_path, 'ly' + str(year) + key + '.npy' ), ly1.astype(np.uint8))



def main():
    years = [2017, 2018]
    path = '/mnt/ds3lab-scratch/lming/data/tiles_brazil/hansen-raw'
    out_path = '/mnt/ds3lab-scratch/lming/data/tiles_brazil/hansen-labels'
    create_dir(out_path)
    for year in years:
        imgs1, imgs2 = get_imgs(path, year)
        label_path = os.path.join(out_path, str(year))
        create_dir(label_path)
        save_labels(imgs1, imgs2, year, label_path)


if __name__=='__main__':
    main()
