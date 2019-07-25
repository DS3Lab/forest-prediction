import os
import glob
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from base import BaseDataLoader
import time

# TODO: png format not implemented

def loadFiles(input_dir, filetype, years=None):
    """
    Assemble dict of file paths.
    :param split: string in ['train', 'val']
    :param input_path: string
    :param years: list of strings or None
    :param instance: bool
    :return: dict
    """
    print('===============YEARS IN LOADFILES==============', years)
    paths_dict = {}
    for path, dirs, files in os.walk(input_dir):
        skip_year = False
        if years is not None:
            current_year = path.rsplit('/', 1)[-1]
            if current_year not in years:
                skip_year = True
        if not skip_year:
            print('Reading from {}'.format(path))
            label_paths, img_paths = searchFiles(path, filetype)
            paths_dict = {**paths_dict, **zipPaths(label_paths, img_paths)}
    if not paths_dict:
        print("WARNING: NOT LOADING ANY FILES")
    return paths_dict


def searchFiles(path, filetype):
    """
    Get file paths via wildcard search.
    :param path: path to files for each city
    :param instance: bool
    :return: 2 lists
    """
    print(filetype,'================')
    assert filetype in ['npy', 'png']
    label_wildcard_search = os.path.join(path, 'ly*.' + filetype)
    label_paths = glob.glob(label_wildcard_search)
    label_paths.sort()
    # img_wildcard_search = os.path.join(path, "ld*.npy")
    img_wildcard_search = os.path.join(path, 'pl*.' + filetype)
    img_paths = glob.glob(img_wildcard_search)
    img_paths.sort()
    return label_paths, img_paths


def zipPaths(label_paths, img_paths):
    """
    zip paths in form of dict.
    :param label_paths: list of strings
    :param img_paths: list of strings
    :return: dict
    """
    try:
        assert len(label_paths) == len(img_paths)
    except:
        raise Exception('Missmatch: {} label paths vs. {} img paths!'.format(len(label_paths), len(img_paths)))

    paths_dict = {}
    for i, img_path in enumerate(img_paths):
        img_spec = ('_').join(img_paths[i].split('/')[-1].split('_'))[:-4]
        try:
            assert img_spec[2:] in label_paths[i][2:]
        except:
            raise Exception('img and label name mismatch: {} vs. {}'.format(img_paths[i], label_paths[i]))
        if img_spec in paths_dict:
            print('zipPaths WARNING', img_spec, paths_dict[img_spec])
        paths_dict[img_spec] = {'img': img_paths[i], 'mask': label_paths[i], 'img_spec': img_spec}
    print('zipPaths', len(label_paths), len(img_paths), len(paths_dict))
    return paths_dict

# TODO: merge with loadFiles
def loadDoubleFiles(input_dir, years=None):
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
            label_paths, img_paths_1, img_paths_2 = searchDoubleFiles(path, current_year)
            paths_dict = {**paths_dict, **zipDoublePaths(label_paths, img_paths_1, img_paths_2)}
    if not paths_dict:
        print("WARNING: NOT LOADING ANY FILES")
    return paths_dict

# TODO: merge with searchFiles
def searchDoubleFiles(path, year):
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

# TODO: merge with zipPaths
def zipDoublePaths(label_paths, img_paths_1, img_paths_2):
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


class PlanetDataset(Dataset):
    """
    Planet 3-month mosaic dataset
    """
    def __init__(self, data_dir,
            years,
            filetype='npy',
            max_dataset_size=float('inf')):
        """Initizalize dataset.
            Params:
                data_dir: absolute path, string
                years: list of years
                filetype: png or npy. If png it is raw data, if npy it has been preprocessed
        """
        start = time.time()
        self.paths_dict = loadFiles(data_dir, filetype, years)
        self.keys = list(self.paths_dict.keys())

        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])

        self.data_size = len(self.paths_dict)
        self.data_dir = data_dir
        # print('Time to log the dataset:', time.time() - start)

    def __len__(self):
        # print('Planet Dataset len called')
        return self.data_size

    def __getitem__(self, index):
        r"""Returns data point and its binary mask"""

        # img_path = self.img_paths[index % self.img_size]  # make sure index is within then range
        # mask_path = self.mask_paths[index % self.mask_size]
        path_dict = self.paths_dict[self.keys[index % self.data_size]]
        img_path = path_dict['img']
        mask_path = path_dict['mask']
        img = torch.from_numpy(np.load(img_path)).float()
        mask = torch.from_numpy(np.load(mask_path)).float().unsqueeze(0)
        return img, mask


class PlanetDataLoader(BaseDataLoader):

    def __init__(self, data_dir,
            batch_size,
            years,
            filetype='npy',
            max_dataset_size=float("inf"),
            shuffle=True,
            num_workers=1,
            training=True):

        if training:
            subdir = os.path.join(data_dir, 'train')
        else:
            subdir = os.path.join(data_dir, 'val')

        self.dataset = PlanetDataset(
                data_dir,
                years,
                filetype,
                max_dataset_size)
        super().__init__(self.dataset, batch_size, shuffle, 0, num_workers)


def open_image(image_path):
    assert image_path[-3:] in ['png', 'npy']
    if filename[-3:] == 'png':
        img_arr = cv2.imread(image_path)
        img_arr = cv2.cvtColor(image_arr, cv2.COLOR_BGR2RGB)  # Change to RGB color ordering
        return np.array(img_arr)
    else:
        return np.load(image_path)


class PlanetImgDataset(Dataset):
    """
    Planet 3-month mosaic dataset
    """
    def __init__(self, data_dir,
            years,
            max_dataset_size=float('inf')):
        """Initizalize dataset.
            Params:
                data_dir: absolute path, string
                years: list of years
                filetype: png or npy. If png it is raw data, if npy it has been preprocessed
        """
        self.paths_dict = loadFiles(data_dir, 'png', years)
        self.keys = list(self.paths_dict.keys())

        # trsfm = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.1307,), (0.3081,))
        # ])

        self.data_size = len(self.paths_dict)
        self.data_dir = data_dir
        # print('Time to log the dataset:', time.time() - start)

    def __len__(self):
        # print('Planet Dataset len called')
        return self.data_size

    def __getitem__(self, index):
        r"""Returns data point and its binary mask"""

        # img_path = self.img_paths[index % self.img_size]  # make sure index is within then range
        # mask_path = self.mask_paths[index % self.mask_size]
        path_dict = self.paths_dict[self.keys[index % self.data_size]]
        img_path = path_dict['img']
        mask_path = path_dict['mask']
        img = torch.from_numpy(np.load(img_path)).float()
        mask = torch.from_numpy(np.load(mask_path)).int().unsqueeze(0)
        return img, mask


class PlanetImgDataLoader(BaseDataLoader):

    def __init__(self, data_dir,
            batch_size,
            years,
            filetype='npy',
            max_dataset_size=float("inf"),
            shuffle=True,
            num_workers=1,
            training=True):

        if training:
            subdir = os.path.join(data_dir, 'train')
        else:
            subdir = os.path.join(data_dir, 'val')

        self.dataset = PlanetImgDataset(
                data_dir,
                years,
                filetype,
                max_dataset_size)
        super().__init__(self.dataset, batch_size, shuffle, 0, num_workers)




'''
def make_dataset(dir, keyname, years, max_dataset_size=float("inf")):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    if years == 'all':
        years = list(next(os.walk(dir))[1])
    if max_dataset_size == 'inf':
        max_dataset_size = float("inf")
    # for subdir in next(os.walk(dir))[1]:
    for year in years:
        images.extend([file for file in glob.glob(os.path.join(dir, year, keyname + '*'))])
    return images
'''
