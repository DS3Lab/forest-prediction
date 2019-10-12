"""This module retrieves a list of the min quality tiles in SA"""
import os
import glob
import pickle as pkl

# PATH = '/mnt/ds3lab-scratch/lming/data/min_quality12/forest_cover_raw'
PATH = '/mnt/ds3lab-scratch/lming/data/min_quality11/forest_cover/processed/2013'
files = glob.glob(os.path.join(PATH, '*.npy'))

tiles = []
for f in files:
    items = f.split('/')[-1].split('_')
    z, x, y = items[1], items[2], items[3][:-4]
    tiles.append((z, x, y))

with open('tiles11.pkl', 'wb') as f:
    pkl.dump(tiles, f)

print(len(tiles))
