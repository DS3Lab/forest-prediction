import os.path
import numpy as np
import torch
from data.base_dataset import BaseDataset, get_transform
from data.numpy_folder import make_planet_dataset, open_image
from PIL import Image
import random


class PlanetDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.
        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        # self.dir = os.path.join(opt.dataroot, opt.phase)
        if opt.phase != 'gen':
            input_dir = os.path.join(opt.dataroot, 'quarter_cropped', opt.phase)
            target_dir = os.path.join(opt.dataroot, 'annual_cropped', opt.phase)
            self.paths = make_planet_dataset(input_dir, target_dir)
        else: # Gen data for video prediction
            self.paths = []
            phases = ['train', 'val', 'test']
            for phase in phases:
                input_dir = os.path.join(opt.dataroot, 'quarter_cropped', phase)
                target_dir = os.path.join(opt.dataroot, 'annual_cropped', phase)
                self.paths.extend(make_planet_dataset(input_dir, target_dir))

        self.size = len(self.paths)

        self.transform_A = get_transform(self.opt, timelapse='quarter')
        self.transform_B = get_transform(self.opt, timelapse='annual')

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path, B_path = self.paths[index % self.size]  # make sure index is within then range
        # A_img = Image.open(A_path).convert('RGB') # png
        # B_img = Image.open(B_path).convert('RGB') # B is already a np array
        # apply image transformation
        A_img = open_image(A_path)
        B_img = open_image(B_path)
        # B_img = torch.from_numpy(open_image(B_path)).double()
        # A = torch.from_numpy(A_img).float()
        # B = torch.from_numpy(B_img).float()
        A = self.transform_A(A_img).float()
        B = self.transform_B(B_img).float()
        # print(A.size(), B.size(), 'GET ITEM SIZE')
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return self.size
