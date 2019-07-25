import os, gdal
import rasterio
import numpy as np

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
    nonzeros = np.where(arr!=0)
    num_losses = nonzeros[0].size
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

lon_lat = [('10N','080W'), ('10N', '070W'),
           ('00N','080W'), ('00N','070W'), 
           ('00N','060W'), ('00N','050W'),
           ('10S','060W'), ('10S','050W'),
           ('20S','070W'), ('20S','060W'), 
           ('30S','070W'),
           ('40S','080W'), ('30S','080W')]
tile_size_x = 224
tile_size_y = 224
n_samples = 400 # samples per granule, 13 granules -> 400*13 = 5200 per year -> >30k
tifs = []
for lon, lat in lon_lat:
    ly2018 = {}
    ly2018['in_path'] = 'hansen/'
    ly2018['input_filename'] = 'lossyear2018_{}_{}.tif'.format(lon, lat)
    ly2018['out_path'] = 'hansen/lossyear/'
    ly2018['output_filename'] = 'ly2018_{}_{}_'
    # Landsat
    ld2018 = {}
    ld2018['in_path'] = 'hansen/'
    ld2018['input_filename'] = 'last2018_{}_{}.tif'.format(lon, lat)
    ld2018['out_path'] = 'hansen/landsat/'
    ld2018['output_filename'] = 'ld2018_{}_{}_'

    ly2017 = {}
    ly2017['in_path'] = 'hansen/'
    ly2017['input_filename'] = 'lossyear2017_{}_{}.tif'.format(lon, lat)
    ly2017['out_path'] = 'hansen/lossyear/'
    ly2017['output_filename'] = 'ly2017_{}_{}_'
    # Landsat
    ld2017 = {}
    ld2017['in_path'] = 'hansen/'
    ld2017['input_filename'] = 'last2017_{}_{}.tif'.format(lon, lat)
    ld2017['out_path'] = 'hansen/landsat/'
    ld2017['output_filename'] = 'ld2017_{}_{}_'

    ly2016 = {}
    ly2016['in_path'] = 'hansen/'
    ly2016['input_filename'] = 'lossyear2016_{}_{}.tif'.format(lon, lat)
    ly2016['out_path'] = 'hansen/lossyear/'
    ly2016['output_filename'] = 'ly2016_{}_{}_'
    # Landsat
    ld2016 = {}
    ld2016['in_path'] = 'hansen/'
    ld2016['input_filename'] = 'last2016_{}_{}.tif'.format(lon, lat)
    ld2016['out_path'] = 'hansen/landsat/'
    ld2016['output_filename'] = 'ld2016_{}_{}_'

    ly2015 = {}
    ly2015['in_path'] = 'hansen/'
    ly2015['input_filename'] = 'lossyear2015_{}_{}.tif'.format(lon, lat)
    ly2015['out_path'] = 'hansen/lossyear/'
    ly2015['output_filename'] = 'ly2015_{}_{}_'
    # Landsat
    ld2015 = {}
    ld2015['in_path'] = 'hansen/'
    ld2015['input_filename'] = 'last2015_{}_{}.tif'.format(lon, lat)
    ld2015['out_path'] = 'hansen/landsat/'
    ld2015['output_filename'] = 'ld2015_{}_{}_'

    ly2014 = {}
    ly2014['in_path'] = 'hansen/'
    ly2014['input_filename'] = 'lossyear2014_{}_{}.tif'.format(lon, lat)
    ly2014['out_path'] = 'hansen/lossyear/'
    ly2014['output_filename'] = 'ly2014_{}_{}_'
    # Landsat
    ld2014 = {}
    ld2014['in_path'] = 'hansen/'
    ld2014['input_filename'] = 'last2014_{}_{}.tif'.format(lon, lat)
    ld2014['out_path'] = 'hansen/landsat/'
    ld2014['output_filename'] = 'ld2014_{}_{}_'

    ly2013 = {}
    ly2013['in_path'] = 'hansen/'
    ly2013['input_filename'] = 'lossyear2013_{}_{}.tif'.format(lon, lat)
    ly2013['out_path'] = 'hansen/lossyear/'
    ly2013['output_filename'] = 'ly2013_{}_{}_'
    # Landsat
    ld2013 = {}
    ld2013['in_path'] = 'hansen/'
    ld2013['input_filename'] = 'last2013_{}_{}.tif'.format(lon, lat)
    ld2013['out_path'] = 'hansen/landsat/'
    ld2013['output_filename'] = 'ld2013_{}_{}_'

    # First year 2000
    ld2000 = {}
    ld2000['in_path'] = 'hansen/'
    ld2000['input_filename'] = 'first2000_{}_{}.tif'.format(lon, lat)
    ld2000['out_path'] = 'hansen/landsat/'
    ld2000['output_filename'] = 'ld2000_{}_{}_'
    tifs.extend([ly2018, ly2017, ly2016, ly2015, ly2014, ly2013, 
                 ld2018, ld2017, ld2016, ld2015, ld2014, ld2013, ld2000])
    # First year 2000
    ld2000 = {}
    ld2000['in_path'] = 'hansen/'
    ld2000['input_filename'] = 'first2000_{}_{}.tif'.format(lon, lat)
    ld2000['out_path'] = 'hansen/landsat/'
    ld2000['output_filename'] = 'ld2000_{}_{}_'
    tifs.extend([ly2018, ly2017, ly2016, ly2015, ly2014, ly2013, 
                 ld2018, ld2017, ld2016, ld2015, ld2014, ld2013, ld2000])

    # Detect in what parts there has been forest loss in 2000-2018. We use Hansen loss-year 2018
    with rasterio.open(os.path.join(ly2018['in_path'], ly2018['input_filename'])) as src:
        b = src.read(1)
        arr = b.transpose() # gdal input is col x row            

    sampleTiles(arr, tile_size_x, tile_size_y, tifs, n_samples)
