import glob
import os

planet_dir = '/mnt/ds3lab-scratch/lming/data/tiles_brazil/train/2018'
new_dir =  '/mnt/ds3lab-scratch/lming/data/tiles_brazil/redo/2018'

mosaics = [file for file in glob.glob(os.path.join(planet_dir, 'pl*'))]

for mosaic in mosaics:
    statinfo = os.stat(mosaic)
    if statinfo.st_size < 10:
        print(mosaic)
        filename = mosaic.split('/')[-1]
        labelname = 'ly' + filename[2:]
        os.rename(mosaic, os.path.join(new_dir, filename))
        os.rename(os.path.join(planet_dir, labelname), os.path.join(new_dir, labelname))

#     print(statinfo.st_size)

# pl2017_12_1539_2097.npy
