import os
import glob
import numpy as np
import cv2
import torch
import torchvision
import pickle as pkl
# import rasterio
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader import utils

# TODO: put in utils
def get_tile_info(tile):
    """
    Retrieve the year, zoom, x, y from a tile. Example: ly2017_12_1223_2516.png
    """
    tile_items = tile.split('_')
    year = tile_items[0][2:]
    z = tile_items[1]
    x = tile_items[2]
    y = tile_items[3][:-4]
    return int(year), z, x, y

# TODO: put in utils
def open_image(img_path):
    filetype = img_path[-3:]
    assert filetype in ['png', 'npy']
    if filetype == 'npy':
        # try:
        img_arr = np.load(img_path)
        if len(img_arr.shape) == 3: # RGB
            if img_arr.shape[0] == 3: # NCHW
                img_arr = img_arr.transpose([1,2,0])
            img_arr = img_arr / 255.
        #     print(img_arr.shape)
            return img_arr
        elif len(img_arr.shape) == 2: # mask
                # change to binary mask
            nonzero = np.where(img_arr!=0)
            img_arr[nonzero] = 1
            return img_arr
        # except:
        #     print('ERROR', img_path)
        #     return None
        # return img_arr / 255.
    else:
        # For images transforms.ToTensor() does range to (0.,1.)

        img_arr = cv2.imread(img_path)
        try:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        except:
            print("ERROR OPENING IMAGE", img_path)
        return cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

def get_img(mask_path, img_dir, double=False):
    year, z, x, y = get_tile_info(mask_path.split('/')[-1])
    if not double:
        if 'planet2landsat' in img_dir:
            img_template = os.path.join(img_dir, str(year), 'pl{year}_{z}_{x}_{y}.png')
        elif 'landsat' in img_dir:
            img_template = os.path.join(img_dir, str(year), 'ld{year}_{z}_{x}_{y}.png')
        else:
            img_template = os.path.join(img_dir, str(year), 'pl{year}_{z}_{x}_{y}.npy')
        return img_template.format(year=year, z=z, x=x, y=y)
    else:
        if 'planet2landsat' in img_dir:
            img_template1 = os.path.join(img_dir, str(year-1), 'pl{year}_{z}_{x}_{y}.png')
            img_template2 = os.path.join(img_dir, str(year), 'pl{year}_{z}_{x}_{y}.png')
        elif 'landsat' in img_dir:
            img_template1 = os.path.join(img_dir, str(year-1), 'ld{year}_{z}_{x}_{y}.png')
            img_template2 = os.path.join(img_dir, str(year), 'ld{year}_{z}_{x}_{y}.png')
        else:
            img_template1 = os.path.join(img_dir, str(year-1), 'pl{year}_{z}_{x}_{y}.npy')
            img_template2 = os.path.join(img_dir, str(year), 'pl{year}_{z}_{x}_{y}.npy')
        return img_template1.format(year=year-1, z=z, x=x, y=y), img_template2.format(year=year, z=z, x=x, y=y)
def get_mask(img_path, mask_dir):
    year, z, x, y = get_tile_info(img_path.split('/')[-1])
    if 'loss' in mask_dir:
        mask_template = os.path.join(mask_dir, str(year), 'fl{year}_{z}_{x}_{y}.npy')
    else: # cover
        mask_template = os.path.join(mask_dir, str(year), 'fc{year}_{z}_{x}_{y}.npy')
    return mask_template.format(year=year, z=z, x=x, y=y)

class PlanetSingleDataset(Dataset):
    """
    Planet 3-month mosaic dataset
    """
    def __init__(self, img_dir, label_dir, years, max_dataset_size, video=False, mode='train'):
        """Initizalize dataset.
            Params:
                data_dir: absolute path, string
                years: list of years
                filetype: png or npy. If png it is raw data, if npy it has been preprocessed
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.paths = []
        # Delete after video training or update dataset properly
        if video:
            with open('/mnt/ds3lab-scratch/lming/forest-prediction/video_prediction/train_val_test.pkl', 'rb') as pkl_file:
                train_val_test = pkl.load(pkl_file)
            for key in train_val_test[mode].keys():
                imgs = train_val_test[mode][key]
                for year in years:
                    mask_path = get_mask(imgs[year], self.label_dir)
                    self.paths.append(mask_path)
        else:
            for year in years:
                imgs_path = os.path.join(label_dir, year)
                self.paths.extend(glob.glob(os.path.join(imgs_path, '*')))
        self.paths = self.paths[:min(len(self.paths), max_dataset_size)]
        self.paths.sort()
        # TODO: update mean/std
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            utils.Normalize((0.3326, 0.3570, 0.2224),
                (0.1059, 0.1086, 0.1283))
        ])
        self.dataset_size = len(self.paths)

    def __len__(self):
        # print('Planet Dataset len called')
        return self.dataset_size

    def __getitem__(self, index):
        r"""Returns data point and its binary mask"""
        # Notes: tiles in annual mosaics need to be divided by 255.
        mask_path = self.paths[index]
        year, z, x, y = get_tile_info(mask_path.split('/')[-1])
        print('PREDICTING TILE {z}_{x}_{y}'.format(z=z, x=x, y=y))
        # For img_dir give
        # /mnt/ds3lab-scratch/lming/data/min_quality11/landsat/min_pct
        img_path = get_img(mask_path, self.img_dir)

        mask_arr = open_image(mask_path)
        img_arr = open_image(img_path)
        mask_arr = torch.from_numpy(mask_arr).unsqueeze(0)
        img_arr = self.transforms(img_arr)

        return img_arr.float(), mask_arr.float()

class PlanetDataLoader(BaseDataLoader):
    def __init__(self, img_dir,
            label_dir,
            batch_size,
            years,
            max_dataset_size=float('inf'),
            shuffle=True,
            num_workers=16,
            video=False,
            mode='train'):
        if max_dataset_size == 'inf':
            max_dataset_size = float('inf')
        self.dataset = PlanetSingleDataset(img_dir, label_dir, years, max_dataset_size, video, mode)
        super().__init__(self.dataset, batch_size, shuffle, 0, num_workers)

class PlanetDoubleDataset(Dataset):
    """
    Planet 3-month mosaic dataset
    """
    def __init__(self, img_dir, label_dir, years, max_dataset_size):
        """Initizalize dataset.
            Params:
                data_dir: absolute path, string
                years: list of years
                filetype: png or npy. If png it is raw data, if npy it has been preprocessed
        """
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.paths = []
        for year in years:
            imgs_path = os.path.join(label_dir, year)
            self.paths.extend(glob.glob(os.path.join(imgs_path, '*')))
        self.paths = self.paths[:min(len(self.paths), max_dataset_size)]
        self.paths.sort()
        # TODO: update mean/std
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            utils.Normalize((0.3326, 0.3570, 0.2224),
                (0.1059, 0.1086, 0.1283))
        ])
        self.dataset_size = len(self.paths)

    def __len__(self):
        # print('Planet Dataset len called')
        return self.dataset_size

    def __getitem__(self, index):
        r"""Returns data point and its binary mask"""
        # Notes: tiles in annual mosaics need to be divided by 255.
        mask_path = self.paths[index]
        year, z, x, y = get_tile_info(mask_path.split('/')[-1])
        img_path1, img_path2 = get_img(mask_path, self.img_dir, double=True)

        mask_arr = open_image(mask_path)
        img_arr1 = open_image(img_path1)
        img_arr2 = open_image(img_path2)
        mask_arr = torch.from_numpy(mask_arr).unsqueeze(0)
        img_arr1 = self.transforms(img_arr1)
        img_arr2 = self.transforms(img_arr2)
        img_arr = torch.cat((img_arr1, img_arr2), 0)
        return img_arr.float(), mask_arr.float()

class PlanetDoubleDataLoader(BaseDataLoader):
    def __init__(self, img_dir,
            label_dir,
            batch_size,
            years,
            max_dataset_size=float('inf'),
            shuffle=True,
            num_workers=16):
        if max_dataset_size == 'inf':
            max_dataset_size = float('inf')
        self.dataset = PlanetDoubleDataset(img_dir, label_dir, years, max_dataset_size)
        super().__init__(self.dataset, batch_size, shuffle, 0, num_workers)

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

class PlanetSingleVideoDataset(Dataset):
    """
    Planet 3-month mosaic dataset
    """
    # def __init__(self, img_dir, label_dir, years, max_dataset_size):
    def __init__(self, img_dir, label_dir, video_dir, max_dataset_size):
        """Initizalize dataset.
            Params:
                filetype: png or npy. If png it is raw data, if npy it has been preprocessed
        """
        self.years = ['2013', '2014', '2015', '2016', '2017']
        # self.img_dir = '/mnt/ds3lab-lming/data/min_quality11/landsat/min_pct'
        # self.video_dir = '/mnt/ds3lab-lming/forest-prediction/video_prediction/landsat_video_prediction_results/ours_deterministic_l1'
        # self.label_dir = '/mnt/ds3lab-lming/data/min_quality11/forest_cover/processed'
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.video_dir = video_dir
        self.paths = get_immediate_subdirectories(self.video_dir)
        # with open('/mnt/ds3lab-scratch/lming/forest-prediction/video_prediction/no_in_training.pkl', 'rb') as f:
        #     no_in_training = pkl.load(f)
        # self.paths = no_in_training
        self.paths.sort()
        print('SELF PATHS', self.paths)
        # self.paths.sort()
        # TODO: update mean/std
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            utils.Normalize((0.3326, 0.3570, 0.2224),
                (0.1059, 0.1086, 0.1283))
        ])
        self.dataset_size = len(self.paths)

    def get_item(self, index):
        key = self.paths[index]
        print('PREDICTING TILE {i} {key}'.format(i=index, key=key))
        img_gt_template = os.path.join(self.img_dir, '{year_dir}', 'ld{year_f}_{key}.png')
        img_video_template = os.path.join(self.video_dir, key, 'gen_image_00000_00_0{}.png')
        label_template = os.path.join(self.label_dir, '{year_dir}', 'fc{year_f}_{key}.npy')

        img2013 = img_gt_template.format(year_dir=2013, year_f=2013, key=key)
        img2014 = img_gt_template.format(year_dir=2014, year_f=2014, key=key)
        img2015 = img_gt_template.format(year_dir=2015, year_f=2015, key=key)
        img2016 = img_gt_template.format(year_dir=2016, year_f=2016, key=key)
        img2017 = img_gt_template.format(year_dir=2017, year_f=2017, key=key)

        pred2015 = img_video_template.format(0)
        pred2016 = img_video_template.format(1)
        pred2017 = img_video_template.format(2)

        label2013 = label_template.format(year_dir=2013, year_f=2013, key=key)
        label2014 = label_template.format(year_dir=2014, year_f=2014, key=key)
        label2015 = label_template.format(year_dir=2015, year_f=2015, key=key)
        label2016 = label_template.format(year_dir=2016, year_f=2016, key=key)
        label2017 = label_template.format(year_dir=2017, year_f=2017, key=key)

        return {
            '2013': {
                'img_dir': img2013,
                'label_dir': label2013
            },
            '2014': {
                'img_dir': img2014,
                'label_dir': label2014
            },
            '2015': {
                'img_dir': img2015,
                'label_dir': label2015
            },
            '2016': {
                'img_dir': img2016,
                'label_dir': label2016
            },
            '2017': {
                'img_dir': img2017,
                'label_dir': label2017
            },
            '2015p': {
                'img_dir': pred2015,
                'label_dir': label2015
            },
            '2016p': {
                'img_dir': pred2016,
                'label_dir': label2016
            },
            '2017p': {
                'img_dir': pred2017,
                'label_dir': label2017
            },
        }

    def _process_img_pair(self, img_dict):
        img_arr = open_image(img_dict['img_dir'])
        mask_arr = open_image(img_dict['label_dir'])
        img_arr = self.transforms(img_arr)
        mask_arr = torch.from_numpy(mask_arr).unsqueeze(0)

        return {
            'img_arr': img_arr.float(),
            'mask_arr': mask_arr.float()
        }

    def __len__(self):
        # print('Planet Dataset len called')
        return self.dataset_size

    def __getitem__(self, index):
        r"""Returns data point and its binary mask"""
        # Notes: tiles in annual mosaics need to be divided by 255.
        imgs_dict = self.get_item(index)
        tensor_dict = {
            '2013': self._process_img_pair(imgs_dict['2013']),
            '2014': self._process_img_pair(imgs_dict['2014']),
            '2015': self._process_img_pair(imgs_dict['2015']),
            '2016': self._process_img_pair(imgs_dict['2016']),
            '2017': self._process_img_pair(imgs_dict['2017']),
            '2015p': self._process_img_pair(imgs_dict['2015p']),
            '2016p': self._process_img_pair(imgs_dict['2016p']),
            '2017p': self._process_img_pair(imgs_dict['2017p']),
        }
        return tensor_dict


class PlanetVideoDataLoader(BaseDataLoader):
    def __init__(self, img_dir,
            label_dir,
            video_dir,
            batch_size,
            max_dataset_size=float('inf'),
            shuffle=False,
            num_workers=16):
        self.dataset = PlanetSingleVideoDataset(img_dir, label_dir, video_dir, max_dataset_size)
        super().__init__(self.dataset, batch_size, shuffle, 0, num_workers)

class PlanetResultsDataset(Dataset):
    """
    Planet 3-month mosaic dataset
    """
    def __init__(self, img_dir, label_dir, years, max_dataset_size, video=False, mode='train'):
        """Initizalize dataset.
            Params:
                data_dir: absolute path, string
                years: list of years
                filetype: png or npy. If png it is raw data, if npy it has been preprocessed
        """
        with open('/mnt/ds3lab-scratch/lming/gee_data/forma_tiles2017.pkl', 'rb') as f:
            self.paths = pkl.load(f)
        # Delete after video training or update dataset properly

        self.paths = self.paths[:min(len(self.paths), max_dataset_size)]
        self.paths.sort()
        # TODO: update mean/std
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            utils.Normalize((0.3326, 0.3570, 0.2224),
                (0.1059, 0.1086, 0.1283))
        ])
        self.dataset_size = len(self.paths)

    def __len__(self):
        # print('Planet Dataset len called')
        return self.dataset_size

    def __getitem__(self, index):
        r"""Returns data point and its binary mask"""
        # Notes: tiles in annual mosaics need to be divided by 255.
        z, x, y = self.paths[index]
        fc_templ = 'fc{year}_{z}_{x}_{y}.npy'
        fl_templ = 'fl{year}_{z}_{x}_{y}.npy'
        ld_templ = 'ld{year}_{z}_{x}_{y}.png'

        fc_path0 = '/mnt/ds3lab-scratch/lming/gee_data/z11/forest_coverv2/2016'
        fc_path1 = '/mnt/ds3lab-scratch/lming/gee_data/z11/forest_coverv2/2017'
        fl_path = '/mnt/ds3lab-scratch/lming/gee_data/z11/forest_lossv2/fl_for_results/'
        ld_path = '/mnt/ds3lab-scratch/lming/gee_data/ldpl/video'
        fc_path0 = os.path.join(fc_path0, fc_templ.format(year='2016', z=z, x=x, y=y))
        fc_path1 = os.path.join(fc_path1, fc_templ.format(year='2017', z=z, x=x, y=y))
        fl_path = os.path.join(fl_path, fl_templ.format(year='2017', z=z, x=x, y=y))
        img_path0 = os.path.join(ld_path, ld_templ.format(year='2016', z=z, x=x, y=y))
        img_path1 = os.path.join(ld_path, ld_templ.format(year='2017', z=z, x=x, y=y))

        fc_arr0 = torch.from_numpy(open_image(fc_path0)).unsqueeze(0)
        fc_arr1 = torch.from_numpy(open_image(fc_path1)).unsqueeze(0)
        fl_arr = torch.from_numpy(open_image(fl_path)).unsqueeze(0)

        img_arr0 = self.transforms(open_image(img_path0))
        img_arr1 = self.transforms(open_image(img_path1))

        return {
            '2016':{
                'img': img_arr0,
                'fc': fc_arr0,
            },
            '2017':{
                'img': img_arr1,
                'fc': fc_arr1
            },
            'fl': fl_arr
        }


class PlanetResultsLoader(BaseDataLoader):
    def __init__(self, img_dir,
            label_dir,
            batch_size,
            years,
            max_dataset_size=float('inf'),
            shuffle=True,
            num_workers=16,
            video=False,
            mode='train'):
        if max_dataset_size == 'inf':
            max_dataset_size = float('inf')
        self.dataset = PlanetResultsDataset(img_dir, label_dir, years, max_dataset_size, video, mode)
        super().__init__(self.dataset, batch_size, shuffle, 0, num_workers)
z
