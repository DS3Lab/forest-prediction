import glob
import os


def loadFiles(input_dir, years=None):
    """
    Assemble dict of file paths.
    :param split: string in ['train', 'val']
    :param input_path: string
    :param years: list of strings or None
    :param instance: bool
    :return: dict
    """
    paths_dict = {}
    for path, dirs, files in os.walk(input_dir):
        skip_year = False
        if years is not None:
            current_year = path.rsplit('/', 1)[-1]
            if current_year not in years:
                skip_year = True
        if not skip_year:
            print('Reading from {}'.format(path))
            label_paths, img_paths_1, img_paths_2 = searchFiles(path, current_year)
            paths_dict = {**paths_dict, **zipPaths(label_paths, img_paths_1, img_paths_2)}
    if not paths_dict:
        print("WARNING: NOT LOADING ANY FILES")
    return paths_dict


def searchFiles(path, year):
    """
    Get file paths via wildcard search.
    :param path: path to files for each city
    :param instance: bool
    :return: 2 lists
    """
    img_paths_1 = glob.glob(os.path.join(path, 'planet', year, '*.npy'))
    img_paths_2 = glob.glob(os.path.join(path, 'planet', str(int(year)-1), '*.npy'))
    label_paths = glob.glob(os.path.join(path, 'hansen', '*.npy'))
    img_paths_1.sort()
    img_paths_2.sort()
    label_paths.sort()
    return label_paths, img_paths_1, img_paths_2


def zipPaths(label_paths, img_paths_1, img_paths_2):
    """
    zip paths in form of dict.
    :param label_paths: list of strings
    :param img_paths: list of strings
    :return: dict
    """
    try:
        assert len(label_paths) == len(img_paths_1) == len(img_paths_2)
    except:
        raise Exception('Missmatch: {} label paths vs. {} img paths!'.format(len(label_paths), len(img_paths_1), len(img_paths_2)))

    paths_dict = {}
    for i, img_path in enumerate(img_paths_1):
        img_spec_1 = ('_').join(img_paths_1[i].split('/')[-1].split('_'))[:-4]
        img_spec_2 = ('_').join(img_paths_2[i].split('/')[-1].split('_'))[:-4]
        label_spec = ('_').join(label_paths[i].split('/')[-1].split('_'))[:-4]
        print(img_spec_1[5:])
        assert img_spec_1[5:] == img_spec_2[5:] == label_spec[5:], \
                'img and label name mismatch: {} vs. {} vs. {}\n {} vs. {} vs. {}'.format(
                        img_paths_1[i], 
                        img_paths_2[i],
                        label_paths[i],
                        img_spec_1,
                        img_spec_2,
                        label_spec)
        paths_dict[img_spec_1] = {
                'img1': img_paths_1[i], 
                'img2': img_paths_2[i],
                "mask": label_paths[i], 
                'img_spec': img_spec_1}
    return paths_dict

