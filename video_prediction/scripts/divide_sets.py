import glob
import os

def reduce_keys(planet_files):
    imgs = {}
    for file in planet_files:
        items = file.split('/')[-1].split('_')
        key = '_'.join((items[1], items[2], items[3][:-4]))
        if key not in imgs:
            imgs[key] = {
                'z': items[1],
                'x': items[2],
                'y': items[3][:-4]
            }
    return imgs

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

def create_dir(folder):
    if not os.path.exists(folder):
        os.makedirs(folder)

def mv_set(src_dir, out_dir, split):
    assert split in ['train', 'val', 'test']
    src_split = os.path.join(src_dir, split)
    out_split = os.path.join(out_dir, split)
    create_dir(out_split)

    files_src = glob.glob(os.path.join(src_split, '*.npy'))
    for file in files_src:
        items = file.split('/')[-1]
        template_name = '_'.join(items[0],'{q}', items[1], items[2], items[3], items[4], items[5][:-4]) + '.png'
        mv_files = [
            template_name.format(q='q1'),
            template_name.format(q='q2'),
            template_name.format(q='q3'),
            template_name.format(q='q4')
        ]
        for file in mv_files:
            print(os.path.join(out_dir, file), os.path.join(out_split, file))
            # os.rename(os.path.join(out_dir, file), os.path.join(out_split, file))
        break

dir_quarter_cropped = 'mnt/ds3lab-scratch/lming/data/min_quality/planet/quarter_croppedv3'
dir_annual_cropped = 'mnt/ds3lab-scratch/lming/data/min_quality/planet/annual_cropped'

mv_set(dir_annual_cropped, dir_quarter_cropped, 'train')
