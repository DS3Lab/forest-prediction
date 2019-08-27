import numpy as np
import glob
import os
import cv2
import logging

def check_quality_label(img, threshold = 0.02):
    if img is None:
        return -1
    count_nonzero = np.count_nonzero(img[:,:,2])  # asume BGR, labels in red channel
    img_size = img[:,:,2].size
    ratio = count_nonzero / img_size
    if ratio >= 0.05:
        return 5
    elif ratio >= 0.04:
        return 4
    elif ratio >= 0.03:
        return 3
    elif ratio >= 0.02:
        return 2
    else:
        return ratio


logger = logging.getLogger('quality-check')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('quality-check.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


img_dir = '/mnt/ds3lab-scratch/lming/data/min_quality/hansen'
five_pct_path = '/mnt/ds3lab-scratch/lming/data/min_quality/hansen/five_pct'
four_pct_path = '/mnt/ds3lab-scratch/lming/data/min_quality/hansen/four_pct'
three_pct_path = '/mnt/ds3lab-scratch/lming/data/min_quality/hansen/three_pct'
two_pct_path = '/mnt/ds3lab-scratch/lming/data/min_quality/hansen/two_pct'

files = glob.glob(os.path.join(img_dir, '*.png'))
files_5pct = glob.glob(os.path.join(five_pct_path, '*.png'))
files_4pct = glob.glob(os.path.join(four_pct_path, '*.png'))
files_3pct = glob.glob(os.path.join(three_pct_path, '*.png'))
files_2pct = glob.glob(os.path.join(two_pct_path, '*.png'))

less_two_pct = 0
two_pct_count = 0
three_pct_count = 0
four_pct_count = 0
five_pct_count = 0

log_every_n_steps = 1000

def check_quality(files, expected_quality):
    for i in range(len(files)):
        logger.debug('')
        file = files[i]
        filename = file.split('/')[-1]
        img = cv2.imread(file)
        quality = check_quality_label(img)
        if quality != expected_quality:
            logger.debug('{} with expected quality {} but found {} instead.'.format(file, expected_quality, quality))

check_quality(files_5pct, 5)
check_quality(files_4pct, 4)
check_quality(files_3pct, 3)
check_quality(files_2pct, 2)

def move_files():
    for i in range(len(files)):  

        if i % log_every_n_steps == 0:
            logger.debug('Logging stats:\n2-3%: {}\n3-4%: {}\n4-5%: {}\n5%+:{}'.format(
			two_pct_count, three_pct_count, four_pct_count, five_pct_count))
        file = files[i]
        filename = file.split('/')[-1]
        img = cv2.imread(file)
        quality = check_quality_label(img)
        if quality == 5:
            outname = os.path.join(five_pct_path, filename)
            five_pct_count += 1
            os.rename(file, outname)
        elif quality == 4:
            outname = os.path.join(four_pct_path, filename)
            four_pct_count += 1
            os.rename(file, outname)
        elif quality == 3:
            outname = os.path.join(three_pct_path, filename)
            three_pct_count += 1
            os.rename(file, outname)
        elif quality == 2:
            outname = os.path.join(two_pct_path, filename)
            two_pct_count += 1
            os.rename(file, outname)
        elif quality == -1:
            logger.debug('Warning: could not open file {}'.format(file))
        else:
            less_two_pct += 1
            logger.debug('Warning: less than 2 detected {} {}'.format(file, quality))
            try:
                os.remove(file)
                logger.debug('File {} deleted'.format(file))
            except:
                logger.debug('Warning, problem deleting file {}'.format(file))


