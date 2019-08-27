import os, gdal
import rasterio
import numpy as np
import glob
import sys
from multiprocessing import Process, Queue
# import multiprocessing

def saveTile(x, y, tile_size_x, tile_size_y, tifs):
    """
    :param x - coordinate x 
    :param y - coordinate y
    :param tile_size - size of the tile
    :param tifs: list of dicts
    """
    print('SAVING TILE', x, y)
    for tif in tifs:
        in_path = tif['in_path']
        input_filename = tif['input_filename']
        out_path = tif['out_path']
        output_filename = tif['output_filename']
        com_string = "gdal_translate -of GTIFF -srcwin " +  \
                      str(x)+ ", " + str(y) + ", " + str(tile_size_x) + ", " + str(tile_size_y) + " " + \
                      str(in_path) + str(input_filename) + " " + \
                      str(out_path) + str(output_filename) + \
                      str(x) + "_" + str(y) + ".tif"
        os.system(com_string)
    
def getTile(arr, tile_size_x, tile_size_y, tifs):
    """
    :param arr - mask where to sample for
    :tifs: list of dicts
    """
    nonzeros = np.where(arr!=0)
    # Tile within boundaries 
    if nonzeros[0].size > 0 and \
    (nonzeros[0][0] + tile_size_x < arr.shape[0]) and \
    (nonzeros[1][0] + tile_size_y < arr.shape[1]):
        # Get first nonzero index, i.e first pixel with HANSEN alert
        x = nonzeros[0][0]
        y = nonzeros[1][0]
        # Copy that part of the matrix
        aux = np.copy(arr[x:x+tile_size_x, y:y+tile_size_y])
        # Update matrix to not sample the same pixels
        arr[x:x+tile_size_x, y:y+tile_size_y] = 0
        saveTile(x, y, tile_size_x, tile_size_y, tifs)
        return aux
    else:
        return None

def sampleTiles(arr, tile_size_x, tile_size_y, 
                tifs, 
                n_samples=None):
    # samples = []
    while(1):
        sample = getTile(arr, tile_size_x, tile_size_y, tifs)
        # print(sample)
        if sample is not None:
            # samples.append(sample)
            pass
        else:
            # return samples
            break
        if n_samples is not None:
            n_samples -= 1
            if n_samples <= 0:
                # return samples
                break

def sampleTiles2(arr, tile_size_x, tile_size_y, tifs, n_samples):
    """This one does random sampling"""
    nonzeros = np.where(arr!=0)
    num_losses = nonzeros[0].size
    print('Sampling {} tiles from {} losses'.format(n_samples, num_losses))
    while n_samples > 0:
        idx = np.random.randint(0, num_losses)
        x = nonzeros[0][idx]
        y = nonzeros[1][idx]
        if nonzeros[0].size > 0 and \
        (x + tile_size_x < arr.shape[0]) and \
        (y + tile_size_y < arr.shape[1]):
            print("Saving tile in", x, y)
            saveTile(x, y, tile_size_x, tile_size_y, tifs)
            n_samples -= 1
        else:
            print("No forest loss found in this raster")
            break 

# tifno = ['1','2','3','4','5','6','7','8','9','10','11','12','13','14','15']
tile_size_x = 224
tile_size_y = 224
# n_samples = 400 # samples per granule, 13 granules -> 400*13 = 5200 per year -> >30k

year_list = sys.argv[1]
years = year_list.split(',')
path = '/mnt/ds3lab-scratch/lming/data/ls7/'

def sample_year_tiles(year, queue=None):
    print('Getting tiles from year', year)
    for r, d, f in os.walk(path + 'original/' + year + '/'):
        for fil in f:
            tifs = []
            ld2018 = {}
            ld2018['in_path'] = path +'original/' + year + '/'
            ld2018['input_filename'] = fil
            ld2018['out_path'] = path + 'raw/' + year + '/'
            ld2018['output_filename'] = 'ld' + year + '_' + fil[:-4] + '_'
            tifs.extend([ld2018])

        # Detect in what parts there has been forest loss in 2000-2018. We use Hansen loss-year 2018
            with rasterio.open(os.path.join(ld2018['in_path'], ld2018['input_filename'])) as src:
                b = src.read(7)
                arr = b.transpose() # gdal input is col x row
            
            if os.path.getsize(os.path.join(ld2018['in_path'], ld2018['input_filename'])) < 500000000:
                n_samples = 50
            elif os.path.getsize(os.path.join(ld2018['in_path'], ld2018['input_filename'])) < 1000000000:
                n_samples = 100
            elif os.path.getsize(os.path.join(ld2018['in_path'], ld2018['input_filename'])) < 1500000000:
                n_samples = 150
            elif os.path.getsize(os.path.join(ld2018['in_path'], ld2018['input_filename'])) < 2000000000:
                n_samples = 300
            else:
                n_samples = 400
            sampleTiles(arr, tile_size_x, tile_size_y, tifs, n_samples)
    if queue is not None:
        queue.put(year)
        return

queue = Queue()
proc = []
for year in years:
    p = Process(target=sample_year_tiles, args=(year, queue))
    p.start()
    proc.append(p)
results = [queue.get() for p in proc]
print('FINISHED', results)




 
# for year in years:
#     print('Getting tiles from year', year)
#     for r, d, f in os.walk(path + 'original/' + year + '/'):
#         for fil in f:
#             tifs = []
#             ld2018 = {}
#             ld2018['in_path'] = path +'original/' + year + '/'
#             ld2018['input_filename'] = fil
#             ld2018['out_path'] = path + 'raw/' + year + '/'
#             ld2018['output_filename'] = 'ld' + year + '_' + fil[:-4] + '_'
#             tifs.extend([ld2018]) 
# 
#         # Detect in what parts there has been forest loss in 2000-2018. We use Hansen loss-year 2018
#             with rasterio.open(os.path.join(ld2018['in_path'], ld2018['input_filename'])) as src:
#                 b = src.read(7)
#                 arr = b.transpose() # gdal input is col x row
#             531000687            
#             if os.path.getsize(os.path.join(ld2018['in_path'], ld2018['input_filename'])) < 500000000:
#                 n_samples = 50
#             elif os.path.getsize(os.path.join(ld2018['in_path'], ld2018['input_filename'])) < 1000000000:
#                 n_samples = 100
#             elif os.path.getsize(os.path.join(ld2018['in_path'], ld2018['input_filename'])) < 1500000000: 
#                 n_samples = 150
#             elif os.path.getsize(os.path.join(ld2018['in_path'], ld2018['input_filename'])) < 2000000000:
#                 n_samples = 200
#             else:
#                 n_samples = 250
#             sampleTiles(arr, tile_size_x, tile_size_y, tifs, n_samples)
'''        
for number in tifno:
    tifs = []
    ld2018 = {}
    ld2018['in_path'] = '/mnt/ds3lab-scratch/lming/data/gee/raw/'
    ld2018['input_filename'] = '{}.tif'.format(number)
    ld2018['out_path'] = '/mnt/ds3lab-scratch/lming/data/gee/output/'
    ld2018['output_filename'] = 'ld2018_'+ number + '_'
    tifs.extend([ld2018]) 

    # Detect in what parts there has been forest loss in 2000-2018. We use Hansen loss-year 2018
    with rasterio.open(os.path.join(ld2018['in_path'], ld2018['input_filename'])) as src:
        b = src.read(4)
        arr = b.transpose() # gdal input is col x row            

    sampleTiles(arr, tile_size_x, tile_size_y, tifs, n_samples)
'''
