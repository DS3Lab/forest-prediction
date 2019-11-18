"""
Module to define the Pytorch Dataset and Dataloaders for training/testing.
Note: the custom DataLoader must inherit from BaseDataLoader
"""
import os
import glob
import numpy as np
import cv2
import torch
import torchvision
import pickle as pkl
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader import utils


class SingleDataset(Dataset):
    """
    Dataset for single image input
        Params:
            img_dir: directory of the input raw images (Planet or Landsat)
            label_dir: directory of the semantic segmentation labels (Hansen)
            years: list of years
            max_dataset_size
    """
    def __init__(self, img_dir, label_dir, years, max_dataset_size):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.paths = []
        for year in years:
            imgs_path = os.path.join(label_dir, year)
            self.paths.extend(glob.glob(os.path.join(imgs_path, '*')))
        self.paths = self.paths[:min(len(self.paths), max_dataset_size)]
        self.paths.sort()
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
        mask_path = self.paths[index]
        img_path = utils.get_img(mask_path, self.img_dir)

        mask_arr = utils.open_image(mask_path)
        img_arr = utils.open_image(img_path)
        mask_arr = torch.from_numpy(mask_arr).unsqueeze(0)
        img_arr = self.transforms(img_arr)

        return img_arr.float(), mask_arr.float()

class SingleDataLoader(BaseDataLoader):
    """
    DataLoader for single image input
        Params:
            img_dir: directory of the input raw images (Planet or Landsat)
            label_dir: directory of the semantic segmentation labels (Hansen)
            years: list of years
    """
    def __init__(self, img_dir,
            label_dir,
            batch_size,
            years,
            max_dataset_size=float('inf'),
            shuffle=True,
            num_workers=16,
            mode='train'):
        if max_dataset_size == 'inf':
            max_dataset_size = float('inf')
        self.dataset = SingleDataset(img_dir, label_dir, years, max_dataset_size)
        super().__init__(self.dataset, batch_size, shuffle, 0, num_workers)

class DoubleDataset(Dataset):
    """
    Dataset for double image input.
        Params:
            img_dir: directory of the input raw images (Planet or Landsat)
            label_dir: directory of the semantic segmentation labels (Hansen)
            years: list of years
    """
    def __init__(self, img_dir, label_dir, years, max_dataset_size):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.paths = []
        for year in years:
            imgs_path = os.path.join(label_dir, year)
            self.paths.extend(glob.glob(os.path.join(imgs_path, '*')))
        self.paths = self.paths[:min(len(self.paths), max_dataset_size)]
        # with open('/mnt/ds3lab-scratch/lming/gee_data/forma_tiles2017.pkl', 'rb') as f:
        #     self.paths = pkl.load(f)
        # self.paths = [('11','773','1071')]
        self.paths.sort()
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            utils.Normalize((0.3326, 0.3570, 0.2224),
                (0.1059, 0.1086, 0.1283))
        ])
        self.dataset_size = len(self.paths)

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        r"""Returns data point and its binary mask"""
        # Notes: tiles in annual mosaics need to be divided by 255.
        '''
        z, x, y = self.paths[index]
        fc_templ = 'fc{year}_{z}_{x}_{y}.npy'
        fl_templ = 'fl{year}_{z}_{x}_{y}.npy'
        ld_templ = 'pl{year}_{z}_{x}_{y}.npy'

        fc_path0 = '/mnt/ds3lab-scratch/lming/gee_data/ldpl/hansen_video/forest_cover/2016'
        fc_path1 = '/mnt/ds3lab-scratch/lming/gee_data/ldpl/hansen_video/forest_cover/2017'
        fl_path = '/mnt/ds3lab-scratch/lming/gee_data/ldpl/hansen_video/forest_loss/2017'
        fl_path = '/mnt/ds3lab-scratch/lming/gee_data/images_forma_compare'
        # ld_path = '/mnt/ds3lab-scratch/lming/gee_data/ldpl/video'
        ld_path = '/mnt/ds3lab-scratch/lming/gee_data/ldpl/planet/annual'
        fl_path = os.path.join(fl_path, fl_templ.format(year='2017', z=z, x=x, y=y))
        img_path0 = os.path.join(ld_path, '2016', ld_templ.format(year='2016', z=z, x=x, y=y))
        img_path1 = os.path.join(ld_path, '2017', ld_templ.format(year='2017', z=z, x=x, y=y))

        fl_arr = torch.from_numpy(open_image(fl_path)).unsqueeze(0)

        img_arr0 = self.transforms(open_image(img_path0))
        img_arr1 = self.transforms(open_image(img_path1))

        img_arr = torch.cat((img_arr0, img_arr1), 0)
        return img_arr.float(), fl_arr.float()
        '''
        mask_path = self.paths[index]
        img_path1, img_path2 = utils.get_img(mask_path, self.img_dir, double=True)

        mask_arr = utils.open_image(mask_path)
        img_arr1 = utils.open_image(img_path1)
        img_arr2 = utils.open_image(img_path2)
        mask_arr = torch.from_numpy(mask_arr).unsqueeze(0)
        img_arr1 = self.transforms(img_arr1)
        img_arr2 = self.transforms(img_arr2)
        img_arr = torch.cat((img_arr1, img_arr2), 0)
        return img_arr.float(), mask_arr.float()


class DoubleDataLoader(BaseDataLoader):
    """
    Data loader for 2 input images
    Params:
        img_dir: directory that contains the raw images. It has subdirectories for
            every year.
        label_dir: directory that contains the labels of the images. It has
            subdirectories for every year.
        years: years to load from img_dir and label_dir.
    """
    def __init__(self, img_dir,
            label_dir,
            batch_size,
            years,
            max_dataset_size=float('inf'),
            shuffle=True,
            num_workers=16):
        if max_dataset_size == 'inf':
            max_dataset_size = float('inf')
        self.dataset = DoubleDataset(img_dir, label_dir, years, max_dataset_size)
        super().__init__(self.dataset, batch_size, shuffle, 0, num_workers)


class VideoDataset(Dataset):
    """
    Dataset to retrieve video prediction results
    Params:
        img_dir: directory that contains the ground truth raw images. It has subdirectories for
            every year.
        label_dir: directory that contains the labels of the images. It has
            subdirectories for every year.
        video_dir: video prediction directory of the images.
        years: years to load from img_dir and label_dir.
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
        self.paths = utils.get_immediate_subdirectories(self.video_dir)
        # with open('/mnt/ds3lab-scratch/lming/forest-prediction/video_prediction/no_in_training.pkl', 'rb') as f:
        #     no_in_training = pkl.load(f)
        # self.paths = no_in_training
        self.paths.sort()
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
        img_arr = utils.open_image(img_dict['img_dir'])
        mask_arr = utils.open_image(img_dict['label_dir'])
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
    """
    DataLoader to retrieve video prediction results
    Params:
        img_dir: directory that contains the ground truth raw images. It has subdirectories for
            every year.
        label_dir: directory that contains the labels of the images. It has
            subdirectories for every year.
        video_dir: video prediction directory of the images.
        years: years to load from img_dir and label_dir.
    """
    def __init__(self, img_dir,
            label_dir,
            video_dir,
            batch_size,
            max_dataset_size=float('inf'),
            shuffle=False,
            num_workers=16):
        self.dataset = VideoDataset(img_dir, label_dir, video_dir, max_dataset_size)
        super().__init__(self.dataset, batch_size, shuffle, 0, num_workers)
