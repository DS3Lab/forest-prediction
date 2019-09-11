import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import requests
import numpy as np

def open_image(img_path):
    filetype = img_path[-3:]
    assert filetype in ['png', 'npy']
    if filetype == 'npy':
        try:
            img_arr = np.load(img_path)
            if len(img_arr.shape) == 3: # RGB
                img_arr = img_arr.transpose([1,2,0])
                return img_arr
            elif len(img_arr.shape) == 2: # mask
                # change to binary mask
                nonzero = np.where(img_arr!=0)
                img_arr[nonzero] = 1
                return img_arr
        except:
            print('ERROR', img_path)
            return None
        # return img_arr / 255.
    else:
        # For images transforms.ToTensor() does range to (0.,1.)
        img_arr = cv2.imread(img_path)
        return cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

def gen_annual_mosaic(q1, q2, q3, q4):
    rq1 = np.copy(q1[:,:,0])
    rq2 = np.copy(q2[:,:,0])
    rq3 = np.copy(q3[:,:,0])
    rq4 = np.copy(q4[:,:,0])

    gq1 = np.copy(q1[:,:,1])
    gq2 = np.copy(q2[:,:,1])
    gq3 = np.copy(q3[:,:,1])
    gq4 = np.copy(q4[:,:,1])

    bq1 = np.copy(q1[:,:,2])
    bq2 = np.copy(q2[:,:,2])
    bq3 = np.copy(q3[:,:,2])
    bq4 = np.copy(q4[:,:,2])
    rq5 = np.dstack((rq1,rq2,rq3,rq4))
    gq5 = np.dstack((gq1,gq2,gq3,gq4))
    bq5 = np.dstack((bq1,bq2,bq3,bq4))

    r = np.median(rq5, axis=2)
    g = np.median(gq5, axis=2)
    b = np.median(bq5, axis=2)

    annual = np.dstack((r,g,b)) # is divided by 255 in the data loader of the video
    return annual

gan_dir = '/mnt/ds3lab-scratch/lming/forest-prediction/video_prediction/results_today/gan/ours_deterministic_l1/'
normal_dir = '/mnt/ds3lab-scratch/lming/forest-prediction/video_prediction/results_today/normalfinal/ours_deterministic_l1/'
img_gan_dir = '/mnt/ds3lab-scratch/lming/data/min_quality/planet/quarter_cropped_gan/test'
img_dir = '/mnt/ds3lab-scratch/lming/data/min_quality/planet/quarter_cropped/test'

gan_dirs = glob.glob(gan_dir + '*/')
normal_dirs = glob.glob(normal_dir + '*/')

out_gan_dir = '/mnt/ds3lab-scratch/lming/forest-prediction/video_prediction/results_today/gan/test'
out_normal_dir = '/mnt/ds3lab-scratch/lming/forest-prediction/video_prediction/results_today/normalfinal/test'
os.makedirs(out_gan_dir)
os.makedirs(out_normal_dir)

out = 'pl{year}_{z}_{x}_{y}_{cx}_{cy}.npy'
pl = 'pl{year}_{q}_{z}_{x}_{y}_{cx}_{cy}.png'

for gan_dir in gan_dirs:
    imgs = glob.glob(os.path.join(gan_dir, '*.png'))
    q2, q3, q4 = imgs[0], imgs[1], imgs[2]
    key = gan_dir.split('/')[1]
    year, z, x, y, cx, cy = key.split('_')
    q1 = os.path.join(img_gan_dir, pl.format(year=year, q='q1', z=z, x=x, y=y, cx=cx, cy=cy))
    annual_mosaic = gen_annual_mosaic(open_image(q1), open_image(q2), open_image(q3), open_image(q4))
    annual_out = os.path.join(out_gan_dir, out.format(year=year, z=z, x=x, y=y, cx=cx, cy=cy))
    np.save(annual_out, annual_mosaic)

for normal_dir in normal_dirs:
    imgs = glob.glob(os.path.join(normal_dir, '*.png'))
    q2, q3, q4 = imgs[0], imgs[1], imgs[2]
    key = normal_dir.split('/')[1]
    year, z, x, y, cx, cy = key.split('_')
    q1 = os.path.join(img_normal_dir, pl.format(year=year, q='q1', z=z, x=x, y=y, cx=cx, cy=cy))
    annual_mosaic = gen_annual_mosaic(open_image(q1), open_image(q2), open_image(q3), open_image(q4))
    annual_out = os.path.join(out_normal_dir, out.format(year=year, z=z, x=x, y=y, cx=cx, cy=cy))
    np.save(annual_out, annual_mosaic)
