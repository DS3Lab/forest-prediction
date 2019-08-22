import glob
import os
import numpy as np

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

PATH = '/mnt/ds3lab-scratch/lming/data/min_quality/hansen_loss'
files = glob.glob(os.path.join(PATH, '*.npy'))

zero_pct = os.path.join(PATH, 'zero_pct') # 0-1
one_pct = os.path.join(PATH, 'one_pct') # 1-2
two_pct = os.path.join(PATH, 'two_pct') # 2-3
three_pct = os.path.join(PATH, 'three_pct') # 3-4
four_pct = os.path.join(PATH, 'four_pct') # 4-5
five_pct = os.path.join(PATH, 'five_pct') # 5+

create_dir(zero_pct)
create_dir(one_pct)
create_dir(two_pct)
create_dir(three_pct)
create_dir(four_pct)
create_dir(five_pct)

for file in files:
    img_arr = np.load(file)
    nonzero_count = np.count_nonzero(img_arr)
    size = img_arr.size
    pct = nonzero_count / size
    if pct <= 0.01:
        out_dir = zero_pct
    elif pct <= 0.02:
        out_dir = one_pct
    elif pct <= 0.03:
        out_dir = two_pct
    elif pct <= 0.04:
        out_dir = three_pct
    elif pct <= 0.05:
        out_dir = four_pct
    else:
        out_dir = five_pct
    name = file.split('/')[-1]
    out_file = os.path.join(out_dir, name)
    os.rename(file, out_file)
