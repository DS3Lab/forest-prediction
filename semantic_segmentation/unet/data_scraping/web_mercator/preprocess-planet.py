"This module overlaps image bands (year and year-1) according to the hansen labels"
import numpy as np
import requests
import rasterio
import os
import glob
import logging
from shutil import copyfile
from tqdm import tqdm


IMGPATH = '/mnt/ds3lab-scratch/lming/data/tiles/planet'
OUTPATH = '/mnt/ds3lab-scratch/lming/data/tiles/input'
YEARS = [2016, 2017, 2018]
logger = logging.getLogger('planet-preprocessing')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('planet-preprocessing.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)


def add_in_dict(dict_, key, value):
    if key in dict_:
        dict_[key].append(value)
    else:
        dict_[key] = [value]


def get_planet_imgs(path, year):
    path1 = os.path.join(path, str(year))
    imgs1 = {}
    for file in glob.glob(path1 + '/*.png'):
        key = '_'.join(file.split('/')[-1][:-4].split('_')[-3:])
        add_in_dict(imgs1, key, file)
    return imgs1

'''
def yearly_mosaics(imgs):
	mosaics = {}
	for key in imgs:
		quads = imgs[key]
		src1 = rasterio.open(quads[0])
		src2 = rasterio.open(quads[1])
		src3 = rasterio.open(quads[2])
		src4 = rasterio.open(quads[3])

		r1 = src1.read(1)
		r2 = src2.read(1)
		r3 = src3.read(1)
		r4 = src4.read(1)
		r5 = np.dstack((r1,r2,r3,r4))

		g1 = src1.read(2)
		g2 = src2.read(2)
		g3 = src3.read(2)
		g4 = src4.read(2)
		g5 = np.dstack((g1,g2,g3,g4))

		b1 = src1.read(3)
		b2 = src2.read(3)
		b3 = src3.read(3)
		b4 = src4.read(3)
		b5 = np.dstack((b1,b2,b3,b4))

		r = np.median(r5, axis=2)
		g = np.median(g5, axis=2)
		b = np.median(b5, axis=2)

		img = np.dstack((r,g,b))
		img = img / 255.
		mosaics[key] = img
	return mosaics

def save_mosaics(mosaics, path, year):
	for key in mosaics:
		np.save(os.path.join(path, 'ly' + year + '_'+ key + '.npy'), mosaics[key])
'''
def gen_yearly_mosaic(quads):
    """
    :params: quads: list of 3-month planet mosaic
    Obtained by the pixel-wise median of the mosaics
    """
    src1 = rasterio.open(quads[0])
    src2 = rasterio.open(quads[1])
    src3 = rasterio.open(quads[2])
    src4 = rasterio.open(quads[3])
    
    r1 = src1.read(1)
    r2 = src2.read(1)
    r3 = src3.read(1)
    r4 = src4.read(1)
    r5 = np.dstack((r1,r2,r3,r4))

    g1 = src1.read(2)
    g2 = src2.read(2)
    g3 = src3.read(2)
    g4 = src4.read(2)
    g5 = np.dstack((g1,g2,g3,g4))

    b1 = src1.read(3)
    b2 = src2.read(3)
    b3 = src3.read(3)
    b4 = src4.read(3)
    b5 = np.dstack((b1,b2,b3,b4))

    r = np.median(r5, axis=2)
    g = np.median(g5, axis=2)
    b = np.median(b5, axis=2)

    yearly_mosaic = np.dstack((r,g,b))
    yearly_mosaic = yearly_mosaic / 255.
    return yearly_mosaic


def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)


def overlap_mosaic(mosaic1, mosaic2):
    """
    Stack together 2 images
    """
    return np.dstack((mosaic1, mosaic2))


def gen_url(mosaic_name):
    MOSAICS_URL = "https://tiles.planet.com/basemaps/v1/planet-tiles/global_quarterly_{year}{q}_mosaic/gmap/{z}/{x}/{y}.png?api_key=25647f4fc88243e2a6e91150aaa117e3"
    splitted = mosaic_name.split('_')
    year, q = splitted[2][:4], splitted[2][4:]
    z=splitted[-3]
    x = splitted[-2]
    y = splitted[-1][:-4]
    url = MOSAICS_URL.format(year=year, q=q, z=z, x=x, y=y)
    return url


def download_item(url, folder, item_type):
    """
    Download quad tif
    """
    if item_type == 'hansen':
        local_filename = '_'.join(url.split('/')[5:])
    elif item_type == 'planet':
        local_filename = '_'.join(url.split('/')[-5:]).split('?')[0]
    # create_dir(folder) # This shouldn't be in the method I think
    print('Downloading', local_filename, 'storing in', os.path.join(folder, local_filename))
    if os.path.isfile(os.path.join(folder, local_filename)):
        return True
    # NOTE the stream=True parameter below
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            if folder:
                path = os.path.join(folder, local_filename)
            else:
                path = local_filename
            with open(path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk: # filter out keep-alive new chunks
                        f.write(chunk)
                        # f.flush()
        logger.debug('SUCCESS: ' + url)
        return True
    except:
        logger.debug('FAILED: ' + url)
        return False



def check_imgs(planet_imgs):
    for img in planet_imgs:
        if not os.path.exists(img):
             url = gen_url(img.split('/')[-1])
             success_download = download_item(url, planet_dir, 'planet')
             if not success_download:
                 log.debug('FAILED TO PROCESS ' + url)
                 return False
    return True


def overlap_annual_mosaics():
    year = '2017'
    # For next time change imgpaths to hansen-labels-min
    # Change out_dir to tiles_brazil/planet-input/year
    # TODO: check the imgpaths that exist (same as here) and move the inputs to another folder
    imgpaths =  [file for file in glob.glob('/mnt/ds3lab-scratch/lming/data/tiles_brazil/hansen-labels-min/{year}/ly*'.format(year=year))]
    planet_dir = '/mnt/ds3lab-scratch/lming/data/tiles_brazil/planet-raw'
    planet_name = 'global_quarterly_{year}{q}_mosaic_gmap_{mosaic}.png'
    out_dir = 'tiles_brazil/hansen-labels-min/{year}'.format(year=year)
    create_dir(out_dir)
    total_imgs = len(imgpaths)
    for i in tqdm(range(len(imgpaths)), disable=False):
        path = imgpaths[i]
        year, mosaic = path.split('/')[-1][2:6], path.split('/')[-1][6:-4]
        filename = 'pl' + year + '_' + mosaic + '.npy'
        out_file = os.path.join(out_dir, filename)
        if os.path.exists(out_file):
            continue
        planet_imgs1 = [
            os.path.join(planet_dir, planet_name.format(year=year, q='q1', mosaic=mosaic)),
            os.path.join(planet_dir, planet_name.format(year=year, q='q2', mosaic=mosaic)),
            os.path.join(planet_dir, planet_name.format(year=year, q='q3', mosaic=mosaic)),
            os.path.join(planet_dir, planet_name.format(year=year, q='q4', mosaic=mosaic))
        ]
        planet_imgs2 = [
            os.path.join(planet_dir, planet_name.format(year=int(year)-1, q='q1', mosaic=mosaic)),
            os.path.join(planet_dir, planet_name.format(year=int(year)-1, q='q2', mosaic=mosaic)),
            os.path.join(planet_dir, planet_name.format(year=int(year)-1, q='q3', mosaic=mosaic)),
            os.path.join(planet_dir, planet_name.format(year=int(year)-1, q='q4', mosaic=mosaic))
        ]
        # Check if the all the images exist
        if check_imgs(planet_imgs1) and check_imgs(planet_imgs2):
            yearly_mosaic1 = gen_yearly_mosaic(planet_imgs1)
            yearly_mosaic2 = gen_yearly_mosaic(planet_imgs2)
            overlapped_mosaic = np.dstack((yearly_mosaic1, yearly_mosaic2)).transpose([2,0,1]) # NCHW
            np.save(out_file, overlapped_mosaic)
            logger.debug('SUCCESS MOSAIC {}'.format(overlapped_mosaic.shape))



def create_3month_dataset():
    year = '2017'
    # For next time change imgpaths to hansen-labels-min
    # Change out_dir to tiles_brazil/planet-input/year
    # TODO: check the imgpaths that exist (same as here) and move the inputs to another folder
    imgpaths =  [file for file in glob.glob('/mnt/ds3lab-scratch/lming/data/tiles_brazil/hansen-labels-min/{year}/ly*'.format(year=year))]
    planet_dir = '/mnt/ds3lab-scratch/lming/data/tiles_brazil/planet-raw'
    planet_name = 'global_quarterly_{year}{q}_mosaic_gmap_{mosaic}.png'
    for i in tqdm(range(len(imgpaths)), disable=False):
        path = imgpaths[i]
        year, mosaic = path.split('/')[-1][2:6], path.split('/')[-1][6:-4]

        filename_q1_q1 = 'pl' + year + '_' + 'q1_q1_' + mosaic + '.npy'
        filename_q1_q2 = 'pl' + year + '_' + 'q1_q2_' + mosaic + '.npy'
        filename_q2_q2 = 'pl' + year + '_' + 'q2_q2_' + mosaic + '.npy'
        filename_q2_q3 = 'pl' + year + '_' + 'q2_q3_' + mosaic + '.npy'
        filename_q3_q3 = 'pl' + year + '_' + 'q3_q3_' + mosaic + '.npy'
        filename_q3_q4 = 'pl' + year + '_' + 'q3_q4_' + mosaic + '.npy'
        filename_q4_q4 = 'pl' + year + '_' + 'q4_q4_' + mosaic + '.npy'

        planet_imgs2 = [
            os.path.join(planet_dir, planet_name.format(year=year, q='q1', mosaic=mosaic)),
            os.path.join(planet_dir, planet_name.format(year=year, q='q2', mosaic=mosaic)),
            os.path.join(planet_dir, planet_name.format(year=year, q='q3', mosaic=mosaic)),
            os.path.join(planet_dir, planet_name.format(year=year, q='q4', mosaic=mosaic))
        ]
        planet_imgs1 = [
            os.path.join(planet_dir, planet_name.format(year=int(year)-1, q='q1', mosaic=mosaic)),
            os.path.join(planet_dir, planet_name.format(year=int(year)-1, q='q2', mosaic=mosaic)),
            os.path.join(planet_dir, planet_name.format(year=int(year)-1, q='q3', mosaic=mosaic)),
            os.path.join(planet_dir, planet_name.format(year=int(year)-1, q='q4', mosaic=mosaic))
        ]

        planet_imgs = [
            (planet_imgs1[0], planet_imgs2[0], filename_q1_q1, 'q1_q1'),  # q1_q1 
            (planet_imgs1[0], planet_imgs2[1], filename_q1_q2, 'q1_q2'),  # q1_q2
            (planet_imgs1[1], planet_imgs2[1], filename_q2_q2, 'q2_q2'),  # q2_q2
            (planet_imgs1[1], planet_imgs2[2], filename_q2_q3, 'q2_q3'),  # q2_q3
            (planet_imgs1[2], planet_imgs2[2], filename_q3_q3, 'q3_q3'),  # q3_q3
            (planet_imgs1[2], planet_imgs2[3], filename_q3_q4, 'q3_q4'),  # q3_q4
            (planet_imgs1[3], planet_imgs2[3], filename_q4_q4, 'q4_q4')   # q4_q4
        ]

        for imgs in planet_imgs:
            filename = imgs[2]
            out_dir = 'tiles_brazil/planet-mosaic-min/{year}_{q}'.format(year=year, q=imgs[3])
            create_dir(out_dir)
            out_file = os.path.join(out_dir, filename)
            if os.path.exists(out_file):
                continue
            img1 = rasterio.open(imgs[0]).read((1,2,3))
            img2 = rasterio.open(imgs[1]).read((1,2,3))
            overlapped_mosaic = np.vstack((img1, img2))
            overlapped_mosaic = overlapped_mosaic / 255.
            np.save(out_file, overlapped_mosaic)
            logger.debug('SUCCESS MOSAIC {}'.format(overlapped_mosaic.shape))


def main():
    year = '2017'
    # For next time change imgpaths to hansen-labels-min
    # Change out_dir to tiles_brazil/planet-input/year
    # TODO: check the imgpaths that exist (same as here) and move the inputs to another folder
    imgpaths =  [file for file in glob.glob('/mnt/ds3lab-scratch/lming/data/tiles_brazil/hansen-labels-min/{year}/ly*'.format(year=year))]
    planet_dir = '/mnt/ds3lab-scratch/lming/data/tiles_brazil/planet-raw'
    planet_name = 'global_quarterly_{year}{q}_mosaic_gmap_{mosaic}.png'
    for i in tqdm(range(len(imgpaths)), disable=False):
        path = imgpaths[i]
        year, mosaic = path.split('/')[-1][2:6], path.split('/')[-1][6:-4]
        qpairs = ['q1_q1','q1_q2','q2_q2','q2_q3',
                   'q3_q3', 'q3_q4', 'q4_q4']       
        for q in qpairs:
            labelname = 'ly' + year + '_' + q + '_' + mosaic + '.npy'
            labelnewdir = os.path.join('tiles_brazil/train', year + '_' + q)
            newpath = os.path.join(labelnewdir, labelname)
            copyfile(path, newpath)
            print('Copied file {} to {}'.format(path, newpath))


if __name__ == '__main__':
	main()

