import os, gdal
import rasterio
import numpy as np
import glob
import sys
import logging
import datetime
import re
from multiprocessing import Process, Queue

GRANULES_PATH = '/mnt/ds3lab-scratch/lming/data/landsat/{year}'
logger = logging.getLogger('tiles')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler('tiles.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

tile_size_x = 256
tile_size_y = 256
n_samples = None

year_list = sys.argv[1]
YEARS = year_list.split(',')

def sample_tiles(args, queue=None):
    """
    args = {
        img_arr: ndarray (7, size_y, size_x)
        tile_shape: int tuple (2,)
        input_filename: original tif file path
        output_filename: ld{year}_{z}_{x}_{y}.npy
        output_labelname: ly{year}_{z}_{x}_{y}.npy
        out_input_path: output dir of landsat tiles
        out_label_path: output dir of hansen tiles
        year: year
    }
    """
    img_arr = args['img_arr'][6]
    H, W = img_arr.shape
    tile_size_x, tile_size_y = args['tile_shape']
    limit_y, limit_x = (H // tile_size_y) * tile_size_y, (W // tile_size_x) * tile_size_x
    logger.debug('======Getting tiles from {}======'.format(args['input_filename']))
    for i in range(0, limit_y, tile_size_y):
        for j in range(0, limit_x, tile_size_x):
            tile = np.copy(img_arr[i:i+tile_size_y, j:j+tile_size_x])
            if check_quality_label(tile):
                save_tile(j, i, args)
    logger.debug('======Finished tiles from {}======'.format(args['input_filename']))
    if queue:
        queue.put(args)

def get_z(granule):
    """granule = 'a/b/c/1.tif"""
    return re.search(r'\d{1,2}', granule.split('/')[-1]).group()

def save_tile(x, y, args):
    year = args['year']
    z = get_z(args['input_filename'])
    output_filename = args['output_filename']
    output_labelname = args['output_labelname']
    out_input_path = args['out_input_path']
    out_label_path = args['out_label_path']
    img_arr0 = np.copy(args['img_arr'][:3, y:y+tile_size_y, x:x+tile_size_x])
    img_arr1 = np.copy(args['img_arr'][3:6, y:y+tile_size_y, x:x+tile_size_x])
    loss = np.copy(args['img_arr'][6, y:y+tile_size_y, x:x+tile_size_x])

    out_file0 = os.path.join(out_input_path, output_filename.format(year=year-1, z=z, x=x, y=y))
    out_file1 = os.path.join(out_input_path, output_filename.format(year=year, z=z, x=x, y=y))
    out_file_loss = os.path.join(out_label_path, output_labelname.format(year=year, z=z, x=x, y=y))

    np.save(out_file0, img_arr0)
    np.save(out_file1, img_arr1)
    np.save(out_file_loss, loss)
    logger.debug('Saving tile {}, {} from {}, '.format(x, y, args['input_filename'], datetime.datetime.now().time()))

def check_quality_label(arr, threshold = 0.02):
    count_nonzero = np.count_nonzero(arr)
    img_size = arr.size
    if count_nonzero / img_size >= threshold:
        return True
    else:
        return False

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def split_list(a_list, num=3):
    fst = len(a_list) // 3
    snd = fst + fst
    return a_list[:fst], a_list[fst:snd], a_list[snd:]

def main():
    out_dir = '/mnt/ds3lab-scratch/lming/data/landsat/min_quality/'
    out_input_path = os.path.join(out_dir, 'landsat')
    out_label_path = os.path.join(out_dir, 'hansen')
    create_dir(out_dir)
    create_dir(out_input_path)
    create_dir(out_label_path)

    for year in YEARS:
        granules_path = GRANULES_PATH.format(year=year)
        granules = glob.glob(os.path.join(granules_path, '*.tif'))
        granule_lists = split_list(granules)
        for granule_list in granule_lists:
            queue = Queue()
            proc = []
            for granule in granule_list:
                with rasterio.open(granule) as dataset:
                    img_arr = dataset.read() # read hansen
                args = {
                    'img_arr': img_arr,
                    'tile_shape': (256, 256),
                    'input_filename': granule,
                    'output_filename': 'ld{year}_{z}_{x}_{y}.npy',
                    'output_labelname': 'ly{year}_{z}_{x}_{y}.npy',
                    'out_input_path':  out_input_path,
                    'out_label_path': out_label_path,
                    'year': int(year)
                }
                p = Process(target=sample_tiles, args=(args, queue))
                p.start()
                proc.append(p)
            results = [queue.get() for p in proc]
                # sample_tiles(args)
                # logger.debug('Finished year {}, granule {}'.format(year, granule))
            print('==============Finished year {}, granule {}'.format(year, granule_list))


if __name__ == '__main__':
    main()
#    for year in years:
#        p = Process(target=sample_tiles, args=(year, queue))
#        p.start()
#        proc.append(p)
#    results = [queue.get() for p in proc]
#    logger.debug('FINISHED', results)


# def save_tile(x, y, tile_size_x, tile_size_y, tif):
#     """
#     :param x - coordinate x
#     :param y - coordinate y
#     :param tile_size - size of the tile
#     :param tifs: list of dicts
#     """
#     granule = tif['granule']
#     logger.debug('Saving tile {}, {} from {}, '.format(x, y, granule, datetime.datetime.now().time()))
#     z = granule.split('/')[-1][:2]
#     out_path = tif['out_path']
#     output_filename = tif['output_filename'].format(
#             year=tif['year'], z=z, x=x, y=y)
#     com_string = "gdal_translate -of GTIFF -srcwin " +  \
#                       str(y)+ ", " + str(x) + ", " + str(tile_size_y) + ", " + str(tile_size_x) + " " + \
#                       granule + " " + \
#                       str(out_path) + str(output_filename)
#     os.system(com_string)

# def get_next_tile(args):
#     """
#     :param arr - mask where to sample for
#     :tifs: list of dicts
#     """
#     tile_size_x = args['tile_size_x']
#     tile_size_y = args['tile_size_y']
#     nonzeros = np.where(args['arr']!=0)
#     # Tile within boundaries
#     if nonzeros[0].size > 0:
#         if nonzeros[0][0] + tile_size_x < args['arr'].shape[0] and \
#         nonzeros[1][0] + tile_size_y < args['arr'].shape[1]:
#             x = nonzeros[0][0]
#             y = nonzeros[1][0]
#             aux = np.copy(args['arr'][x:x+tile_size_x, y:y+tile_size_y])
#             args['arr'][x:x+tile_size_x, y:y+tile_size_y] = 0
#             if check_quality_label(aux):
#                 save_tile(x, y, tile_size_x, tile_size_y, args)
#         return True
#     return False
#     """
#     if nonzeros[0].size > 0 and \
#     (nonzeros[0][0] + tile_size_x < args['arr'].shape[0]) and \
#     (nonzeros[1][0] + tile_size_y < args['arr'].shape[1]):
#         # Get first nonzero index, i.e first pixel with HANSEN alert
#         x = nonzeros[0][0]
#         y = nonzeros[1][0]
#         # Copy that part of the matrix
#         aux = np.copy(args['arr'][x:x+tile_size_x, y:y+tile_size_y])
#         # Update matrix to not sample the same pixels
#         args['arr'][x:x+tile_size_x, y:y+tile_size_y] = 0
#         if check_quality_label(aux):
#             save_tile(x, y, tile_size_x, tile_size_y, args)
#             return True
#     return False
    # """

# def sample_tiles(args, queue=None):
#     # tile_size_x = args['tile_size_x']
#     # tile_size_y = args['tile_size_y']
#     # arr = args['arr']
#     # tif = args['tif']
#     n_samples = None
#     # samples = []
#     i_curr = 1
#     i_next = 0
#     while(1):
#         # sample = get_next_tile(arr, tile_size_x, tile_size_y, tif)
#         sample = get_next_tile(args)
#         # print(sample)
#         if sample:
#             # samples.append(sample)
#             i_curr = i_curr + 1
#             pass
#         else:
#             # return samples
#             return 0
#         if n_samples is not None:
#             n_samples -= 1
#             if n_samples <= 0:
#                 # return samples
#                 return 0
#         i_next = i_next + 1
#         if i_next >= i_curr:
#             return -1
#
#     if queue:
#         queue.put(args)
