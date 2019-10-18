import os
import glob
import numpy as np
from utils import create_dir
from shutil import copyfile

# path = '/mnt/ds3lab-scratch/lming/gee_data/z11/forest_loss'
path = '/mnt/ds3lab-scratch/lming/gee_data/z11/forest_lossv2'
# years = ['2013', '2014', '2015', '2016', '2016_1', '2017', '2017_1']
years = ['2018']

outpath = os.path.join(path, 'twoplus')
create_dir(outpath)
stats = {}

def check_quality_label(img):
    if img is None:
        return -1
    count_nonzero = np.count_nonzero(img)  # asume BGR, labels in red channel
    img_size = img.size
    ratio = count_nonzero / img_size
    if ratio >= 0.05:
        return 5
    elif ratio >= 0.04:
        return 4
    elif ratio >= 0.03:
        return 3
    elif ratio >= 0.02:
        return 2
    elif ratio >= 0.01:
        return 1
    else:
        return 0

for year in years:
    year_path = os.path.join(path, year)
    year_images = glob.glob(os.path.join(year_path, '*.npy'))
    create_dir(os.path.join(outpath, year))
    print('Analizing', year, len(year_images))
    p0 = 0
    p1 = 0
    p2 = 0
    p3 = 0
    p4 = 0
    p5 = 0
    for img in year_images:
        q = check_quality_label(np.load(img))
        if q > 1:
            out_img = os.path.join(outpath, year, img.split('/')[-1])
            copyfile(img, out_img)
        if q == 1:
            p1 += 1
        elif q == 2:
            p2 += 1
        elif q == 3:
            p3 += 1
        elif q == 4:
            p4 += 1
        elif q==5:
            p5 += 1
        else:
            p0 += 1
    stats[year] = {
        'p0': p0,
        'p1': p1,
        'p2': p2,
        'p3': p3,
        'p4': p4,
        'p5': p5
    }

print(stats)
