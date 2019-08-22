import os
import glob
import logging


logger = logging.getLogger('check-tiles')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
fh = logging.FileHandler('check-tiles.log')
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

source_dir = '/mnt/ds3lab-scratch/lming/data/min_quality'
planet_dir = os.path.join(source_dir, 'planet')
hansen_dir = os.path.join(source_dir, 'hansen')

hansen_files = glob.glob(hansen_dir + '/*.png')
for file in hansen_files:
    file_name = file.split('/')[-1]
    name_split = file_name.split('_')
    year = int(name_split[0][2:])
    zoom = name_split[1]
    x = name_split[2]
    y = name_split[3][:-4]
    name_template = 'pl' + '{year}_{q}_' + zoom + '_' + x + '_' + y + '.png'
    planet_template = os.path.join(planet_dir, name_template)
    quads = [
        planet_template.format(year=year, q='q1'),
        planet_template.format(year=year, q='q2'),
        planet_template.format(year=year, q='q3'),
        planet_template.format(year=year, q='q4'),
        planet_template.format(year=year-1, q='q1'),
        planet_template.format(year=year-1, q='q2'),
        planet_template.format(year=year-1, q='q3'),
        planet_template.format(year=year-1, q='q4')
    ]
    for quad in quads:
        if not os.path.isfile(quad):
            logger.debug('re-download: {}'.format(quad))
