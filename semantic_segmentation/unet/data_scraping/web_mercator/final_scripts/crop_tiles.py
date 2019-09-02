import shutil
import os
import glob
import image_slicer
import logging

SRC_PATH = '/mnt/ds3lab-scratch/lming/data/min_quality/planet'
src_quarter_path = os.path.join(SRC_PATH, 'quarter')
out_path = os.path.join(SRC_PATH, 'quarter_cropped')

logger = logging.getLogger('crop')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('crop.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

def get_prefix(name):
    return name.split('/')[-1][:-4]

def main():
    images = glob.glob(os.path.join(src_quarter_path, '*.png'))
    for image in images:
        try:
            prefix = get_prefix(image)
            tiles = image_slicer.slice(image, 16, save=False) # 16 for 64x64 tiles
            image_slicer.save_tiles(tiles, directory=out_path, prefix=prefix)
            logger.debug('SUCCESS: ' + image)
        except:
            logger.debug('FAILED: ' + image)

if __name__ == '__main__':
    main()
