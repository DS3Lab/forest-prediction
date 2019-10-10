"""This module retrieves a list of the min quality tiles in SA"""
import os
import glob
import pickle as pkl

PATH = '/mnt/ds3lab-scratch/lming/data/min_quality12/forest_cover_raw'
files = glob.glob(os.path.join(PATH, '*.png'))

tiles = []
for f in files:
    items = f.split('/')[-1].split('_')
    z, x, y = items[1], items[2], items[3][:-4]
    tiles.append((z, x, y))

with open('tiles.pkl', 'wb') as f:
    pkl.dump(tiles, f)

print(len(tiles))
