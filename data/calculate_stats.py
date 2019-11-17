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
    """
    Open image as ndarray
    """
    filetype = img_path[-3:]
    assert filetype in ['png', 'npy']
    if filetype == 'npy':
        try:
            img_arr = np.load(img_path)
            if len(img_arr.shape) == 3: # RGB
                # if RGB, it expects a (3, height, width)
                # transpose it to (height, width, 3)
                img_arr = img_arr.transpose([1,2,0])
                return img_arr / 255.
            elif len(img_arr.shape) == 2: # mask
                # change to binary mask
                nonzero = np.where(img_arr!=0)
                img_arr[nonzero] = 1
                return img_arr
        except:
            print('ERROR', img_path)
            return None
        # return img_arr / 255.
    else:
        # For images transforms.ToTensor() does range to (0.,1.)
        img_arr = cv2.imread(img_path)
        try:
            img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        except:
            print(img_path)
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
    def __init__(self, img_paths):
        """Initizalize dataset.
            Params:
                img_paths: list of image files of the dataset.
        """
        self.dataset = img_paths
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
        img = open_image(self.dataset[index])
        img = self.transforms(img)
        # print(img)
        return img

def main():
    img_paths = []
    label_dir = '/mnt/ds3lab-scratch/lming/data/min_quality11/landsat/min_pct'
    years = ['2013', '2014', '2015']
    for year in years:
        imgs = glob.glob(os.path.join(label_dir, year, '*'))
        img_paths.extend(imgs)

    dataset = MyDataset(img_paths)
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
    with open('landsat_stats.pkl', 'wb') as pkl_file:
        pkl.dump(stats, pkl_file)

if __name__ == '__main__':
    main()
