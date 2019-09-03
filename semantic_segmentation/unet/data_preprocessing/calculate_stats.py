"""
This script is used to calculate the online mean and std of the dataset
"""

import os
import glob
import numpy as np
import cv2
import torch
import torchvision
import pickle as pkl
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def open_image(img_path):
    filetype = img_path[-3:]
    assert filetype in ['png', 'npy']
    if filetype == 'npy':
        # (3,256,256)->(256,256,3). With transforms.ToTensor()
        # it will be converted to NCHW format.
        return np.load(img_path).transpose([1,2,0]) / 255.
    else:
        img_arr = cv2.imread(img_path)
        return cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

def online_mean_and_std(loader):
    """
        Compute the mean and sd in an online fashion [1].
        Var[x] = E[X^2] - E^2[X]
        [1] https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/9
    """
    cnt = 0
    fst_moment = torch.empty(3, dtype=torch.double)
    snd_moment = torch.empty(3, dtype=torch.double)
    i = 0
    for data in loader:
        b, c, h, w = data.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(data, dim=[0, 2, 3]).double()
        sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3]).double()
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)
        cnt += nb_pixels
        if i % 5000 == 0:
            print(i)
        i = i + 1

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


class MyDataset(Dataset):
    """
    Custom dataset to calculate the mean of annual & quarter mosaics
    """
    def __init__(self, data_dir):
        """Initizalize dataset.
            Params:
                data_dir: absolute path, string
                years: list of years
                filetype: png or npy. If png it is raw data, if npy it has been preprocessed
        """
        # if timelapse == 'quarter':
        #     self.data_dir = '/mnt/ds3lab-scratch/lming/data/min_quality/planet/quarter'
        # else:
        #     self.data_dir = '/mnt/ds3lab-scratch/lming/data/min_quality/planet/annual'

        self.dataset = glob.glob(os.path.join(self.data_dir, 'pl2016*'))
        self.dataset.extend(glob.glob(os.path.join(self.data_dir, 'pl2017*')))

        self.dataset_size = len(self.dataset)
        print('Loaded {} files'.format(self.dataset_size))
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
        ])

    def __len__(self):
        # print('Planet Dataset len called')
        return self.dataset_size

    def __getitem__(self, index):
        r"""Returns data point and its binary mask"""

        # img_path = self.img_paths[index % self.img_size]  # make sure index is within then range
        # mask_path = self.mask_paths[index % self.mask_size]
        # TODO - use paths_dict
        img = open_image(self.dataset[index])
        img = self.transforms(img)
        return img

def main():
    dataset = MyDataset('/mnt/ds3lab-scratch/lming/data/min_quality/planet/quarter_cropped/train')
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False
    )

    mean, std = online_mean_and_std(loader)
    stats = {
        'mean': mean,
        'std': std
    }
    with open('cropped_quarter_stats.pkl', 'wb') as pkl_file:
        pkl.dump(stats, pkl_file)

if __name__ == '__main__':
    main()
