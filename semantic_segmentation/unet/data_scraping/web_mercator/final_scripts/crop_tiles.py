import shutil
import os
import glob
import image_slicer
import logging
from multiprocessing import Process

logger = logging.getLogger('crop')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('crop.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

def get_prefix(name):
    return name.split('/')[-1][:-4]

def getListOfFiles(dirName):
    # create a list of file and sub directories
    # names in the given directory
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)

    return allFiles

def get_hansen_quality_files(dirname='/mnt/ds3lab-scratch/lming/data/min_quality/forest_loss_yearly'):
    # Get images with hansen loss > 1%
    # Omit 0-1% because otherwise there are too many images, too much time to train (time restriction)
    # hansen_one = os.path.join(dirname, 'one_pct')
    hansen_two = os.path.join(dirname, 'two_pct')
    hansen_three = os.path.join(dirname, 'three_pct')
    hansen_four = os.path.join(dirname, 'four_pct')
    hansen_five = os.path.join(dirname, 'five_pct')
    # files_one = getListOfFiles(hansen_one)
    files_two = getListOfFiles(hansen_two)
    files_three = getListOfFiles(hansen_three)
    files_four = getListOfFiles(hansen_four)
    files_five = getListOfFiles(hansen_five)
    return files_three + files_four + files_five

def get_planet_files(hansen_files, planet_dir):
    planet_files = []
    planet_template = 'pl{year}_{q}_{z}_{x}_{y}.png'
    for file in hansen_files:
        file_name = file.split('/')[-1]
        name_split = file_name.split('_')
        year = name_split[0][2:]
        zoom = name_split[1]
        x = name_split[2]
        y = name_split[3][:-4]
        files = [
            os.path.join(planet_dir, planet_template.format(year=year, q='q1', z=zoom, x=x, y=y)),
            os.path.join(planet_dir, planet_template.format(year=year, q='q2', z=zoom, x=x, y=y)),
            os.path.join(planet_dir, planet_template.format(year=year, q='q3', z=zoom, x=x, y=y)),
            os.path.join(planet_dir, planet_template.format(year=year, q='q4', z=zoom, x=x, y=y))
        ]
        planet_files.extend(files)
    return planet_files

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def blockshaped(arr, nrows, ncols):
    """
    Return an array of shape (n, nrows, ncols) where
    n * nrows * ncols = arr.size

    If arr is a 2D array, the returned array should look like n subblocks with
    each subblock preserving the "physical" layout of arr.
    """
    h, w = arr.shape
    assert h % nrows == 0, "{} rows is not evenly divisble by {}".format(h, nrows)
    assert w % ncols == 0, "{} cols is not evenly divisble by {}".format(w, ncols)
    return (arr.reshape(h//nrows, nrows, -1, ncols)
               .swapaxes(1,2)
               .reshape(-1, nrows, ncols))

def get_forest_cover_files(prefixes, forest_dir):
    forest_template = 'gf2000_{z_x_y}.png'
    forest_files = []
    for prefix in prefixes:
        forest_files.append(os.path.join(forest_dir, forest_template.format(z_x_y=prefix[7:])))
    return forest_files

def get_forest_gain_files(prefixes, forest_dir):
    forest_template = 'fg2012_{z_x_y}.png'
    forest_files = []
    for prefix in prefixes:
        forest_files.append(os.path.join(forest_dir, forest_template.format(z_x_y=prefix[7:])))
    return forest_files

def try_to_append(filename, dir_, list_):
    pcts = ['five_pct', 'four_pct', 'three_pct', 'two_pct']
    fileyear = filename[2:6]
    if fileyear == '2017':
        year = '2017'
    else:
        year = '2018'
    for pct in pcts:
        file_abs_path = os.path.join(dir_, 'hansen', pct, year, filename)
        if os.path.exists(file_abs_path):
            list_.append(file_abs_path)

        file_abs_path2 = os.path.join(dir_, 'hansen_other', filename)

        if os.path.exists(file_abs_path2):
            list_.append(file_abs_path2)

def get_forest_loss_files(prefixes, forest_loss_dir):
    loss_template = '{}.png'
    forest_files = []
    for prefix in prefixes:
        try_to_append(loss_template.format(prefix), forest_loss_dir, forest_files)
    return forest_files

def split_images(args):
    for image in args['images']:
        try:
            prefix = get_prefix(image)
            # print('GOING TO SLICE', prefix, args['out_dir'], image)
            tiles = image_slicer.slice(image, 16, save=False)
            image_slicer.save_tiles(tiles, directory=args['out_dir'], prefix=prefix)
            logger.debug('SUCCESS:' + image)
        except:
            logger.debug('FAILED:' + image)

def check_duplicate_keys(forest_cover_files):
    keys = []
    duplicated=0
    for file in forest_cover_files:
        f = file.split('/')[-1][7:19]
        if f in keys:
            print('DUPLICATED', f)
            duplicated += 1
        keys.append(f)
    print('total duplicated', duplicated)


def main():
    SRC_PATH = '/mnt/ds3lab-scratch/lming/data/min_quality'
    FC_PATH = os.path.join(SRC_PATH, 'forest_cover_raw')
    FL_PATH = os.path.join(SRC_PATH, 'hansen')
    out_fc_path = os.path.join(SRC_PATH, 'forest_cover_raw_cropped')
    out_fl_path = os.path.join(SRC_PATH, 'forest_loss_raw_cropped')
    out_fg_path = os.path.join(SRC_PATH, 'forest_gain_raw_cropped')
    create_dir(out_fc_path)
    create_dir(out_fl_path)
    create_dir(out_fg_path)
    hansen_files = get_hansen_quality_files()
    # planet_files = get_planet_files(hansen_files, src_quarter_path)
    # print(len(planet_files), len(hansen_files))
    prefixes = [get_prefix(image) for image in hansen_files]
    forest_loss_files = get_forest_loss_files(prefixes, SRC_PATH)
    forest_cover_files = get_forest_cover_files(prefixes, FC_PATH)
    forest_gain_files = get_forest_gain_files(prefixes, os.path.join(SRC_PATH, 'forest_gain'))
    
    print('FOREST LOSS FILES')
    print(forest_loss_files[:10])
    print('FOREST COVER FILES')
    print(forest_cover_files[:10])
    print(len(forest_cover_files), len(forest_loss_files), len(hansen_files))
    print(len(forest_gain_files))
    print(prefixes[:10])
    # NOTE: THERE ARE LESS FILES IN FOREST GAIN BECAUSE THOSE TILES DIDNT EXIST
    # check_duplicate_keys(forest_gain_files)
    # split_images({'images': forest_gain_files, 'out_dir': out_fg_path})
    #Pros = []
    #p1 = Process(target=split_images, args=({'images': forest_loss_files, 'out_dir': out_fl_path},))
    #p2 = Process(target=split_images, args=({'images': forest_cover_files, 'out_dir': out_fc_path},))
    #Pros.append(p1)
    #Pros.append(p2)
    #p1.start()
    #p2.start()

    #for t in Pros:
    #    t.join()

    #for image in hansen_files:
    #    try:
    #        prefix = get_prefix(image)
    #        print(prefix)
            # tiles = image_slicer.slice(image, 16, save=False) # 16 for 64x64 tiles
            # image_slicer.save_tiles(tiles, directory=out_path, prefix=prefix)
    #        logger.debug('SUCCESS: ' + image)
    #    except:
    #        logger.debug('FAILED: ' + image)

    # for image in planet_files:
    #     try:
    #         prefix = get_prefix(image)
    #         tiles = image_slicer.slice(image, 16, save=False) # 16 for 64x64 tiles
    #         image_slicer.save_tiles(tiles, directory=out_path, prefix=prefix)
    #         logger.debug('SUCCESS: ' + image)
    #     except:
    #         logger.debug('FAILED: ' + image)

if __name__ == '__main__':
    main()
