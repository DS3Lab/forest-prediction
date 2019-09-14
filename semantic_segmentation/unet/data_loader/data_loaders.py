import os
import glob
import numpy as np
import cv2
import torch
import torchvision
# import rasterio
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms
from base import BaseDataLoader
from data_loader import utils3m

HANSEN_PATH_DB = '/mnt/ds3lab-scratch/lming/data/min_quality/forest_cover_processed/no_pct/nips'
PLANET_PATH_DB = '/mnt/ds3lab-scratch/lming/data/min_quality/planet/forest_cover_3m_nips17'
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
        try:
            img_arr = np.load(img_path)
            if len(img_arr.shape) == 3: # RGB
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

def get_item(hansen_file):
    year, z, x, y = get_tile_info(hansen_file.split('/')[-1])
    tile_template = os.path.join(PLANET_PATH_DB, 'pl2017_{q}_16_{x}_{y}.png')
    tiles = utils3m.zoom2tiles(int(z), x, y, 16)
    path_dict = {}
    for tile in tiles:
        key = str(tile[0]) + '_' + str(tile[1])
        path_dict[key] = {
            'q1': tile_template.format(q='q1', x=tile[0], y=tile[1]),
            'q2': tile_template.format(q='q1', x=tile[0], y=tile[1]),
            'q3': tile_template.format(q='q1', x=tile[0], y=tile[1]),
            'q4': tile_template.format(q='q1', x=tile[0], y=tile[1])
        }
    assert len(path_dict) == 256
    return path_dict


class PlanetSingleDataset(Dataset):
    """
    Planet 3-month mosaic dataset
    """
    def __init__(self):
        """Initizalize dataset.
            Params:
                data_dir: absolute path, string
                years: list of years
                filetype: png or npy. If png it is raw data, if npy it has been preprocessed
        """
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            Normalize((0.2311, 0.2838, 0.1752),
                (0.1265, 0.0955, 0.0891))
        ])
        self.paths_dict = glob.glob(os.path.join(HANSEN_PATH_DB, '*'))
        print('IMAGE PATHS', self.paths_dict, len(self.paths_dict))
        self.dataset_size = len(self.paths_dict)

    def __len__(self):
        # print('Planet Dataset len called')
        return self.dataset_size

    def __getitem__(self, index):
        r"""Returns data point and its binary mask"""
        # Notes: tiles in annual mosaics need to be divided by 255.
        mask_path = self.paths_dict[index]
        year, z, x, y = get_tile_info(mask_path.split('/')[-1])

        path_dict = get_item(mask_path)
        keys = list(path_dict.keys())
        img_dict = {}

        # Original mask, zoom=12
        mask_arr = open_image(mask_path)
        # Upsampled mask, zoom=16
        big_mask_arr = utils3m.upsample_tile(12, 16, mask_arr)
        # z16 tile info
        beg_x, beg_y, num_tiles = utils3m.zoom2zoom(int(z), x, y, 16)

        for key in keys:
            tile_x, tile_y = key.split('_')
            quarter_dict = path_dict[key]
            img_dict[key] = {}
            # q1, q2, q3, q4 = open_image(quarter_dict['q1']), \
            #     open_image(quarter_dict['q2']), \
            #     open_image(quarter_dict['q3']), \
            #     open_image(quarter_dict['q4'])
            # annual = utils3m.gen_annual_mosaic(q1, q2, q3, q4)
            mask = utils3m.big2small_tile(big_mask_arr, int(beg_x), int(beg_y), int(tile_x), int(tile_y))
            # Transform to tensor
            mask = torch.from_numpy(mask).unsqueeze(0)
            # q1, q2, q3, q4, annual = self.transforms(q1), \
            #     self.transforms(q2), \
            #     self.transforms(q3), \
            #     self.transforms(q4), \
            #     self.transforms(annual)
            q4 = self.transforms(open_image(quarter_dict['q4']))
            # annual = self.transforms(annual)
            # img_dict[key]['q1'] = q1
            # img_dict[key]['q2'] = q2
            # img_dict[key]['q3'] = q3
            # img_dict[key]['q4'] = q4
            img_dict[key]['annual'] = q4
            img_dict[key]['mask'] = mask

        # Transform masks
        mask_arr = torch.from_numpy(mask_arr).unsqueeze(0)
        big_mask_arr = torch.from_numpy(big_mask_arr).unsqueeze(0)

        img_dict['mask_arr'] = mask_arr
        img_dict['big_mask'] = big_mask_arr
        img_dict['key'] = torch.tensor([int(x), int(y)])
        return img_dict
        # img_dict = get_item(self.paths_dict[index])
        #
        # img_arr0 = self.transforms(open_image(path_dict['img'][0]))
        # img_arr1 = self.transforms(open_image(path_dict['img'][1]))
        # img_arr2 = self.transforms(open_image(path_dict['img'][2]))
        # img_arr3 = self.transforms(open_image(path_dict['img'][3]))
        # img_arr_q1_q2 = torch.cat((img_arr0, img_arr1), 0)
        # img_arr_q2_q3 = torch.cat((img_arr1, img_arr2), 0)
        # img_arr_q3_q4 = torch.cat((img_arr2, img_arr3), 0)
        # mask_arr = open_image(path_dict['mask'])
        # mask_arr = torch.from_numpy(mask_arr).unsqueeze(0)
        # return {
        #     'imgs': (img_arr_q1_q2, img_arr_q2_q3, img_arr_q3_q4),
        #     'mask': mask_arr
        # }

class PlanetDataLoader(BaseDataLoader):
    def __init__(self, input_dir,
            label_dir,
            batch_size,
            years,
            qualities,
            timelapse,
            max_dataset_size=float('inf'),
            shuffle=True,
            num_workers=1,
            testing=False,
            training=True,
            quarter_type='same_year',
            source='planet',
            input_type='two',
            img_mode='same'):
        self.dataset = PlanetSingleDataset()
        super().__init__(self.dataset, batch_size, shuffle, 0, num_workers)

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
