import shutil
import os
import glob
import image_slicer

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
    return files_two + files_three + files_four + files_five

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

def main():
    SRC_PATH = '/mnt/ds3lab-scratch/lming/data/min_quality/planet'
    src_quarter_path = os.path.join(SRC_PATH, 'quarter')
    out_path = os.path.join(SRC_PATH, 'quarter_croppedv2')

    hansen_files = get_hansen_quality_files()
    planet_files = get_planet_files(hansen_files, src_quarter_path)

    for image in images:
        prefix = get_prefix(image)
        tiles = image_slicer.slice(image, 16, save=False)
        image_slicer.save_tiles(tiles, directory=out_path, prefix=prefix)

if __name__ == '__main__':
    main()
