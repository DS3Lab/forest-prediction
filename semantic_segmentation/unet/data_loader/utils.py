"""
Module to add helper functions for data_loaders.py
"""
import torch
import torchvision
import os
import numpy as np
import cv2

def get_immediate_subdirectories(a_dir):
    """Get the immediate subdirectories from a directory
    """
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

def get_tile_info(tile):
    """
    Retrieve the year, zoom, x, y from a tile name.
    Format example: ly2017_12_1223_2516.png
    """
    tile_items = tile.split('_')
    year = tile_items[0][2:]
    z = tile_items[1]
    x = tile_items[2]
    y = tile_items[3][:-4]
    return int(year), z, x, y


# TODO: put in utils
def open_image(img_path):
    """
    Return ndarray from an image of format png or npy
    """
    filetype = img_path[-3:]
    assert filetype in ['png', 'npy']
    if filetype == 'npy':
        img_arr = np.load(img_path)
        if len(img_arr.shape) == 3: # RGB
            if img_arr.shape[0] == 3: # NCHW
                img_arr = img_arr.transpose([1,2,0])
            img_arr = img_arr / 255.
            return img_arr
        elif len(img_arr.shape) == 2: # mask
            # transform to binary mask
            nonzero = np.where(img_arr!=0)
            img_arr[nonzero] = 1
            return img_arr
    else:
        # For images transforms.ToTensor() does range to (0.,1.)
        img_arr = cv2.imread(img_path)
        # img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        return cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)


def get_img(mask_path, img_dir, double=False):
    """
    Get the image template format wrt the filepath
    """
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
    """
    Get the mask template format wrt the filepath
    """
    year, z, x, y = get_tile_info(img_path.split('/')[-1])
    if 'loss' in mask_dir:
        mask_template = os.path.join(mask_dir, str(year), 'fl{year}_{z}_{x}_{y}.npy')
    else: # cover
        mask_template = os.path.join(mask_dir, str(year), 'fc{year}_{z}_{x}_{y}.npy')
    return mask_template.format(year=year, z=z, x=x, y=y)


class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    .. note::
        This transform acts out of place, i.e., it does not mutates the input tensor.

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    [1]. https://pytorch.org/docs/stable/_modules/torchvision/transforms/transforms.html#Normalize
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        return normalize(tensor.double(), self.mean, self.std, self.inplace)


    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


def _is_tensor_image(img):
        return torch.is_tensor(img) and img.ndimension() == 3


def normalize(tensor, mean, std, inplace=False):
    """Normalize a tensor image with mean and standard deviation.
    .. note::
        This transform acts out of place by default, i.e., it does not mutates the input tensor.
    See :class:`~torchvision.transforms.Normalize` for more details.
    Args:
        tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation inplace.
    Returns:
        Tensor: Normalized Tensor image.
    """
    if not _is_tensor_image(tensor):
        raise TypeError('tensor is not a torch image.')

    if not inplace:
        tensor = tensor.clone()

    dtype = tensor.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor
