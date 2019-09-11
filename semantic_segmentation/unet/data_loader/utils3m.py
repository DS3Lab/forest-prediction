import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import requests
import numpy as np
import skimage.measure
import rasterio

def zoom2zoom(z0, x0, y0, z1):
    """
    Get the corresponding tiles from z0_x0_y0 with zoom z1
    Right now: z0=12, z1=16
    Return: new upper left x,y of z1, and number of subsequent tiles in one direction
    """
    z0 = int(z0)
    x0 = int(x0)
    y0 = int(y0)
    z1 = int(z1)
    assert z0 < z1
    zoom_dif = z1 -z0
    x1 = 2**zoom_dif*x0 # corresponding x from the same upper left in zoom z1 coordinate
    y1 = 2**zoom_dif*y0 # corresponding y from the same upper left in zoom z1 coordinate
    num_tiles = 2**zoom_dif
    return x1, y1, num_tiles

def zoom2tiles(z0, x0, y0, z1):
    z0 = int(z0)
    x0 = int(x0)
    y0 = int(y0)
    z1 = int(z1)
    new_x, new_y, num_tiles = zoom2zoom(int(z0), int(x0), int(y0), int(z1))
    tiles = []
    for i in range(num_tiles):
        for j in range(num_tiles):
            tiles.append((new_x+i, new_y+j))
    return tiles

def upsample_tile(z0, z1, img_arr):
    # Assume img_arr is binary -> shape = (256, 256)
    assert z0 < z1
    zoom_dif = z1 - z0
    size_x, size_y = img_arr.shape
    new_size_x, new_size_y = 2**zoom_dif*size_x, 2**zoom_dif*size_y
    res = cv2.resize(img_arr, dsize=(new_size_x, new_size_y), interpolation=cv2.INTER_NEAREST)
    return res

def tile2coord(beg_x, beg_y, tile_x, tile_y, img_size=(256,256)):
    '''
    Tile z=12 is splitted in 256 tiles of z=16
    Tile z then is zoomed to have size 256*16 = 4096
    From a small tile (one of the 256 tiles), retrieve the corresponding coordinates of the 4096x4096 big tile
    '''
    size_x, size_y = img_size
    return (tile_x - beg_x) * size_x, (tile_y - beg_y) * size_y

def big2small_tile(big_tile, beg_x, beg_y, tile_x, tile_y, img_size=(256,256)):
    '''
    Tile z=12 is splitted in 256 tiles of z=16
    Tile z then is zoomed to have size 256*16 = 4096
    From a small tile (one of the 256 tiles), retrieve the corresponding 256x256 that maps the mini tile of z=16
    '''
    '''big_tile: np.array'''
    size_x, size_y = img_size
    big_x, big_y = tile2coord(beg_x, beg_y, tile_x, tile_y, img_size)
    return np.copy(big_tile[big_x:big_x+size_x, big_y:big_y+size_y])

def downsample_tile(z0, z1, img_arr):
    assert z0 < z1
    k, s = 2**(z1-z0), 2**(z1-z0)
    downsample = skimage.measure.block_reduce(img_arr, (k,s), np.mean)
    idxs = downsample >= 0.5
    tile = np.zeros(downsample.shape)
    tile[idxs] = 1
    return tile

def reconstruct_tile(tile_dict, beg_x, beg_y, num_tiles):
    """
    tile_dict{
        key: img_arr
    }
    """
    row_tiles = {}
    print('DEBUGGING RECONSTRUCT TILE', len(tile_dict))
    # Init empty arrays
    for i in range(beg_x, beg_x + num_tiles):
        row_tiles[str(i)] = []

    for i in range(beg_x, beg_x + num_tiles):
        for j in range(beg_y, beg_y + num_tiles):
            key = str(i) + '_' + str(j)
            row_tiles[str(i)].append(tile_dict[key])
    # Merge arrays
    rows = []
    for i in range(beg_x, beg_x + num_tiles):
        row = np.hstack(row_tiles[str(i)])
        plt.figure()
        plt.title(str(i))
        plt.imshow(row)
        rows.append(row)
    return np.vstack(rows)

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

    annual = np.dstack((r,g,b)) / 255.
    return annual
