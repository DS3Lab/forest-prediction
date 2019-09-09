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

HANSEN_PATH_DB = '/mnt/ds3lab-scratch/lming/data/min_quality/hansen_loss_db'

# TODO: put in utils
def loadFiles(mask_dir, img_dir, years, qualities,
    timelapse='quarter',limit=float('inf'), testing=False,
    quarter_type='same_year'):
    """
    Search for the img masks and stores them in a list

    :param input_dir: Directory where the images are stored.
    :param type: 'quarter' or 'year'
        Dataset format:
        Project/
        |-- input_dir/
        |   |-- hansen/
        |   |   |-- five_pct/
        |   |   |   |-- 2018/
        |   |   |   |   |-- *.png
        |   |   |   |-- 2017/
        |   |   |   |-- ...
        |   |   |-- four_pct/
        |   |   |   |-- 2018/
        |   |   |   |-- 2017/
        |   |   |   |-- ...
        |   |-- planet/
        |   |   |   |-- *.png
    :param years: Years of tiles to be loaded
    """
    print('LOAD FILES', timelapse)
    imgs = {}
    for quality in qualities:
        for year in years:
            masks_path = os.path.join(mask_dir, quality, year)
            mask_imgs = glob.glob(os.path.join(masks_path, '*')) ## all files
            print('LOAD FILES MASK IMGS', len(mask_imgs), masks_path)
            for mask in mask_imgs:
                if timelapse == 'quarter':
                    mask_dict = get_quarter_imgs_from_mask(mask, img_dir, testing, quarter_type)
                else:
                    mask_dict = get_annual_imgs_from_mask(mask, img_dir)
                # print('LOAD FILES', mask_dict)
                imgs = {**imgs, **mask_dict}
            if len(imgs) > limit: # soft limitsdasdsad, returns when it is greater
                imgs = {k: imgs[k] for k in list(imgs)[:limit]}
                print('LOAD FILES', len(imgs))
                return imgs
    print('LOAD FILES', len(imgs))
    return imgs

# TODO: put in utils
def loadSingleFiles(mask_dir, img_dir, years, qualities,
    timelapse='quarter',limit=float('inf'), testing=False,
    img_mode='same'):
    """
    img_mode: same - gets only one year
              cont - year - 1 and curr year
    """
    print('LOAD FILES', timelapse)
    imgs = {}
    for quality in qualities:
        for year in years:
            masks_path = os.path.join(mask_dir, quality, year)
            mask_imgs = glob.glob(os.path.join(masks_path, '*')) ## all files
            print('LOAD FILES MASK IMGS', len(mask_imgs), masks_path)
            for mask in mask_imgs:
                if timelapse == 'quarter':
                    mask_dict = get_single_quarter_imgs_from_mask(mask, img_dir, img_mode)
                else:
                    mask_dict = get_annual_imgs_from_mask(mask, img_dir, True, img_mode)
                # print('LOAD FILES', mask_dict)
                imgs = {**imgs, **mask_dict}
            if len(imgs) > limit: # soft limitsdasdsad, returns when it is greater
                imgs = {k: imgs[k] for k in list(imgs)[:limit]}
                print('LOAD FILES', len(imgs))
                return imgs
    print('LOAD FILES', len(imgs))
    return imgs

# TODO: put in utils
def get_tile_info(tile):
    """
    Retrieve the year, zoom, x, y from a tile. Example: ly2017_12_1223_2516.png
    """
    items = tile.split('_')
    year, zoom, x, y, cx, cy = items[0][2:], items[1], items[2], items[3], items[4], items[5][:-4]
    return int(year), zoom, x, y, cx, cy

# TODO: put get_imgs together.
# TODO: put another field to choose between source image type, e.g landsat /planet
def get_annual_imgs_from_mask(mask_file, img_dir, single=False, img_mode='same'):
    """
    Retrieve the annual input images from a mask.
    """
    year, z, x, y, cx, cy = get_tile_info(mask_file.split('/')[-1])
    key = str(year) + '_' + z + '_' + x + '_' + y + '_' + cx + '_' + cy
    planet_name = 'pl' + '{year}' + '_{z}_{x}_{y}_{cx}_{cy}.npy'
    planet_template = os.path.join(img_dir, planet_name)

    data = {}
    if not single:
        data[key] = {
            'img': (planet_template.format(year=year-1, z=z, x=x, y=y, cx=cx, cy=cy),
                    planet_template.format(year=year, z=z, x=x, y=y, cx=cx, cy=cy)),
            'mask': mask_file
        }
    else:
        if img_mode == 'same':
            data[key] = {
                'img': planet_template.format(year=year, z=z, x=x, y=y, cx=cx, cy=cy),
                'mask': mask_file
            }
        else: # cont, return image year - 1, image year
            mask_name = 'fc{year}_{z}_{x}_{y}_{cx}_{cy}.npy'.format(
                year=year-1, z=z, x=x, y=y, cx=cx, cy=cy
            )
            mask_dir = '/'.join(mask_file.split('/')[:-2])
            mask0 = os.path.join(mask_dir, str(year-1), mask_name)
            data[key] = {
                'img': (planet_template.format(year=year-1, z=z, x=x, y=y, cx=cx, cy=cy),
                        planet_template.format(year=year, z=z, x=x, y=y, cx=cx, cy=cy)),
                'mask': (mask0, mask_file),
                'loss': os.path.join(HANSEN_PATH_DB,
                'ly{year}_{z}_{x}_{y}.npy'.format(year=year, z=z, x=x, y=y, cx=cx, cy=cy))
            }
    return data

def get_single_quarter_imgs_from_mask(mask_file, img_dir, img_mode):
    year, z, x, y, cx, cy = get_tile_info(mask_file.split('/')[-1])
    planet_name = 'pl' + '{year}' + '_{q}_{z}_{x}_{y}_{cx}_{cy}.png'
    planet_template = os.path.join(img_dir, planet_name)

    data = {}
    key = str(year) + '_' + z + '_' + x + '_' + y
    # Put the same the quads together so it can be retrieved and plotted
    # at the same time while testing
    data[key] = {
        'img': (planet_template.format(year=year-1, q='q1', z=z, x=x, y=y, cx=cx, cy=cy),
                planet_template.format(year=year-1, q='q2', z=z, x=x, y=y, cx=cx, cy=cy),
                planet_template.format(year=year-1, q='q3', z=z, x=x, y=y, cx=cx, cy=cy),
                planet_template.format(year=year-1, q='q4', z=z, x=x, y=y, cx=cx, cy=cy)),
        'mask': mask_file,
        'loss': os.path.join(HANSEN_PATH_DB,
        'ly{year}_{z}_{x}_{y}_{cx}_{cy}.npy'.format(year=year, z=z, x=x, y=y, cx=cx, cy=cy))
    }

    return data

def get_quarter_imgs_from_mask(mask_file, img_dir, testing=False, quarter_type='same_year'):
    """
    Input img from year t - 1 quarter 1 to quarter 2, loss from year t.
    :param quarter_type: ['same', 'dif']
        'same_year': takes q1 and q2 from the same year
        'next_year': takes q1 from year t-1 and q2 from year t
    """
    assert quarter_type in ['same_year', 'next_year']
    year, z, x, y, cx, cy = get_tile_info(mask_file.split('/')[-1])
    planet_name = 'pl' + '{year}' + '_{q}_{z}_{x}_{y}_{cx}_{cy}.png'
    planet_template = os.path.join(img_dir, planet_name)

    data = {}
    if testing:
        key = str(year) + '_' + z + '_' + x + '_' + y
        # Put the same the quads together so it can be retrieved and plotted
        # at the same time while testing
        data[key] = {
            'img': (planet_template.format(year=year-1, q='q1', z=z, x=x, y=y, cx=cx, cy=cy),
                    planet_template.format(year=year-1, q='q2', z=z, x=x, y=y, cx=cx, cy=cy),
                    planet_template.format(year=year-1, q='q3', z=z, x=x, y=y, cx=cx, cy=cy),
                    planet_template.format(year=year-1, q='q4', z=z, x=x, y=y, cx=cx, cy=cy)),
            'mask': mask_file
        }
    else:
        key = str(year) + '_{q}_' + z + '_' + x + '_' + y
        quarters = ['q1_q2', 'q2_q3', 'q3_q4']
        for quarter in quarters:
            if quarter_type == 'same_year':
                img0 = planet_template.format(year=year-1, q=quarter[:2], z=z, x=x, y=y, cx=cx, cy=cy)
                img1 = planet_template.format(year=year-1, q=quarter[-2:], z=z, x=x, y=y, cx=cx, cy=cy)
                img = (img0, img1)
            else: # next_year
                img0 = planet_template.format(year=year-1, q=quarter[:2], z=z, x=x, y=y, cx=cx, cy=cy)
                img1 = planet_template.format(year=year, q=quarter[-2:], z=z, x=x, y=y, cx=cx, cy=cy)
                img = (img0, img1)

            data[key.format(q=quarter)] = {
                'img':img,
                'mask': mask_file
            }
    # print("Get quarter imgs from mask", data)
    return data

# TODO: put in utils
def open_image(img_path):
    filetype = img_path[-3:]
    assert filetype in ['png', 'npy']
    if filetype == 'npy':
        # annual_mosaics has shape (3,256,256), range [0,255]
        # change to shape (256,256,3) and range [0.,1.]
        # Apparently transforms.ToTensor() doesn't range float numpys, so it has
        # to be ranged here.
        try:
        #     print('OPEN IMAGE',img_path)
            img_arr = np.load(img_path)
            if len(img_arr.shape) == 3: # RGB
                img_arr = img_arr.transpose([1,2,0])
                return img_arr / 255.
            elif len(img_arr.shape) == 2: # mask
                # change to binary mask
                nonzero = np.where(img_arr!=0)
                img_arr[nonzero] = 1
                return img_arr
            print('ERROR', img_path)
        except:
            print('ERROR', img_path)
            return None
        # return img_arr / 255.
    else:
        # For images transforms.ToTensor() does range to (0.,1.)
        img_arr = cv2.imread(img_path)
        return cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)


class PlanetDataset(Dataset):
    """
    Planet 3-month mosaic dataset
    """
    def __init__(self, input_dir, label_dir,
            years,
            qualities,
            timelapse,
            max_dataset_size=float('inf'),
            training=True,
            testing=False,
            quarter_type='same_year'):
        """Initizalize dataset.
            Params:
                data_dir: absolute path, string
                years: list of years
                filetype: png or npy. If png it is raw data, if npy it has been preprocessed
        """
        if training:
            assert testing == False
        elif testing:
            assert training == False

        self.testing = testing
        self.img_dir = input_dir
        self.mask_dir = label_dir
        # self.mask_dir = os.path.join(data_dir, 'hansen')
        # self.img_dir = os.path.join(data_dir, 'planet')
        self.timelapse = timelapse
        if self.timelapse == 'quarter':
            self.img_dir = os.path.join(self.img_dir, 'quarter_cropped')
            self.transforms = transforms.Compose([
                # transforms.RandomHorizontalFlip(), # try later, only works on imgs
                transforms.ToTensor(),
                Normalize((0.2397, 0.2852, 0.1837),
                    (0.1685, 0.1414, 0.1353))
            ])
        else: # annual
            self.img_dir = os.path.join(self.img_dir, 'annual_cropped')
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                Normalize((0.2311, 0.2838, 0.1752),
                    (0.1265, 0.0955, 0.0891))
            ])

        if max_dataset_size == 'inf':
            max_dataset_size = float('inf')
        print('LOAD FILES INIT qualities', qualities)
        print('LOAD FILES INIT timelapse', timelapse)
        self.paths_dict = loadFiles(self.mask_dir, self.img_dir,
            years, qualities, timelapse, max_dataset_size, testing,
            quarter_type)

        self.keys = list(self.paths_dict.keys())
        self.dataset_size = len(self.paths_dict)


    def __len__(self):
        # print('Planet Dataset len called')
        return self.dataset_size

    def __getitem__(self, index):
        r"""Returns data point and its binary mask"""
        # Notes: tiles in annual mosaics need to be divided by 255.
        #
        key = self.keys[index % self.dataset_size]
        path_dict = self.paths_dict[key]
        if not self.testing or self.timelapse == 'annual':
            img_arr0 = open_image(path_dict['img'][0]).astype(np.float64)
            img_arr1 = open_image(path_dict['img'][1]).astype(np.float64)
            mask_arr = open_image(path_dict['mask'])
            # mask_arr = open_mask(path_dict['mask']).astype(np.uint8)
            # mask_arr = transforms.ToTensor()(mask_arr)
            mask_arr = torch.from_numpy(mask_arr).unsqueeze(0)
            # Take only R channel (labels) and transform to tensor
            # mask_arr = transforms.ToTensor()(mask_arr[:,:,0])
            img_arr0 = self.transforms(img_arr0)
            img_arr1 = self.transforms(img_arr1)
            img_arr =  torch.cat((img_arr0, img_arr1), 0) # concatenate images
            # img_arr =  torch.cat((img_tensor0, img_tensor1), 0) # concatenate images
#           print('Final shapes', img_arr.size(), mask_arr.size())
            return img_arr.float(), mask_arr.float()
        else:
            img_arr0 = self.transforms(open_image(path_dict['img'][0]))
            img_arr1 = self.transforms(open_image(path_dict['img'][1]))
            img_arr2 = self.transforms(open_image(path_dict['img'][2]))
            img_arr3 = self.transforms(open_image(path_dict['img'][3]))
            img_arr_q1_q2 = torch.cat((img_arr0, img_arr1), 0)
            img_arr_q2_q3 = torch.cat((img_arr1, img_arr2), 0)
            img_arr_q3_q4 = torch.cat((img_arr2, img_arr3), 0)
            mask_arr = open_image(path_dict['mask'])
            mask_arr = torch.from_numpy(mask_arr).unsqueeze(0)
            # mask_arr = transforms.ToTensor()(mask_arr[:,:,0])

            return {
                'imgs': (img_arr_q1_q2, img_arr_q2_q3, img_arr_q3_q4),
                'mask': mask_arr
            }

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
        assert source in ['planet', 'landsat']
        assert input_type in ['one', 'two'] # 2 images as input or 1 image as input
        assert img_mode in ['same', 'cont'] #
        # if training:
        #     subdir = os.path.join(data_dir, 'train')
        # else:
        #     subdir = os.path.join(data_dir, 'val')
        if input_type == 'two':
            if source == 'planet':
                self.dataset = PlanetDataset(
                        input_dir,
                        label_dir,
                        years,
                        qualities,
                        timelapse,
                        max_dataset_size,
                        training,
                        testing,
                        quarter_type)
                print('USING PLANET DOUBLE DATASET')
            else: # source == landsat
                self.dataset = PlanetTifDataset(input_dir,
                        max_dataset_size=float('inf'))
        else: # one
            self.dataset = PlanetSingleDataset(
                input_dir,
                label_dir,
                years,
                qualities,
                timelapse,
                max_dataset_size,
                training,
                testing,
                img_mode)
            print('USING PLANET SINGLE DATASET')
        super().__init__(self.dataset, batch_size, shuffle, 0, num_workers)

# TODO: move to utils, probably only the function normalize is needed
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


class PlanetSingleDataset(Dataset):
    """
    Planet 3-month mosaic dataset
    """
    def __init__(self, input_dir, label_dir,
            years,
            qualities,
            timelapse,
            max_dataset_size=float('inf'),
            training=True,
            testing=False,
            quarter_type='same_year'):
        """Initizalize dataset.
            Params:
                data_dir: absolute path, string
                years: list of years
                filetype: png or npy. If png it is raw data, if npy it has been preprocessed
        """
        if training:
            assert testing == False
        elif testing:
            assert training == False

        self.testing = testing
        self.img_dir = input_dir
        self.mask_dir = label_dir
        self.timelapse = timelapse
        if self.timelapse == 'quarter_cropped':
            self.img_dir = os.path.join(self.img_dir, 'quarter_cropped')
            self.transforms = transforms.Compose([
                # transforms.RandomHorizontalFlip(), # try later, only works on imgs
                transforms.ToTensor(),
                Normalize((0.2397, 0.2852, 0.1837),
                    (0.1685, 0.1414, 0.1353))
            ])
        else: # annual
            self.img_dir = os.path.join(self.img_dir, 'annual_cropped')
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                Normalize((0.2311, 0.2838, 0.1752),
                    (0.1265, 0.0955, 0.0891))
            ])

        if max_dataset_size == 'inf':
            max_dataset_size = float('inf')
        print('LOAD FILES INIT qualities', qualities)
        print('LOAD FILES INIT timelapse', timelapse)
        self.paths_dict = loadSingleFiles(self.mask_dir, self.img_dir,
            years, qualities, timelapse, max_dataset_size, testing,
            quarter_type)

        self.keys = list(self.paths_dict.keys())
        self.dataset_size = len(self.paths_dict)


    def __len__(self):
        # print('Planet Dataset len called')
        return self.dataset_size

    def __getitem__(self, index):
        r"""Returns data point and its binary mask"""
        # Notes: tiles in annual mosaics need to be divided by 255.
        #
        key = self.keys[index % self.dataset_size]
        path_dict = self.paths_dict[key]

        if not self.testing or self.timelapse == 'annual_cropped':
            img_arr = open_image(path_dict['img']).astype(np.float64)
            mask_arr = open_image(path_dict['mask'])
            mask_arr = torch.from_numpy(mask_arr).unsqueeze(0)
            # mask_arr = transforms.ToTensor()(mask_arr)
            img_arr = self.transforms(img_arr)
            return img_arr.float(), mask_arr.float()
        else:
            img_arr0 = self.transforms(open_image(path_dict['img'][0]))
            img_arr1 = self.transforms(open_image(path_dict['img'][1]))
            img_arr2 = self.transforms(open_image(path_dict['img'][2]))
            img_arr3 = self.transforms(open_image(path_dict['img'][3]))
            mask_arr = open_image(path_dict['mask'])
            mask_arr = torch.from_numpy(mask_arr).unsqueeze(0)
            return {
                'imgs': (img_arr0, img_arr1, img_arr2, img_arr3),
                'mask': mask_arr
            }

class PlanetSingleDataset(Dataset):
    """
    Planet 3-month mosaic dataset
    """
    def __init__(self, input_dir, label_dir,
            years,
            qualities,
            timelapse,
            max_dataset_size=float('inf'),
            training=True,
            testing=False,
            img_mode='same'):
        """Initizalize dataset.
            Params:
                data_dir: absolute path, string
                years: list of years
                filetype: png or npy. If png it is raw data, if npy it has been preprocessed
        """
        if training:
            assert testing == False
        elif testing:
            assert training == False

        self.testing = testing
        self.img_dir = input_dir
        self.mask_dir = label_dir
        self.timelapse = timelapse
        self.img_mode = img_mode
        if self.timelapse == 'quarter':
            self.img_dir = os.path.join(self.img_dir, 'quarter_cropped')
            self.transforms = transforms.Compose([
                # transforms.RandomHorizontalFlip(), # try later, only works on imgs
                transforms.ToTensor(),
                Normalize((0.2397, 0.2852, 0.1837),
                    (0.1685, 0.1414, 0.1353))
            ])
        else: # annual
            self.img_dir = os.path.join(self.img_dir, 'annual_cropped')
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                Normalize((0.2311, 0.2838, 0.1752),
                    (0.1265, 0.0955, 0.0891))
            ])

        if max_dataset_size == 'inf':
            max_dataset_size = float('inf')
        print('LOAD FILES INIT qualities', qualities)
        print('LOAD FILES INIT timelapse', timelapse)
        self.paths_dict = loadSingleFiles(self.mask_dir, self.img_dir,
            years, qualities, timelapse, max_dataset_size, testing,
            img_mode)

        self.keys = list(self.paths_dict.keys())
        self.dataset_size = len(self.paths_dict)


    def __len__(self):
        # print('Planet Dataset len called')
        return self.dataset_size

    def __getitem__(self, index):
        r"""Returns data point and its binary mask"""
        # Notes: tiles in annual mosaics need to be divided by 255.
        #
        key = self.keys[index % self.dataset_size]
        path_dict = self.paths_dict[key]

        if not self.testing or self.timelapse == 'annual_cropped':
            if self.img_mode == 'same':
                img_arr = open_image(path_dict['img']).astype(np.float64)
                mask_arr = open_image(path_dict['mask'])
                mask_arr = torch.from_numpy(mask_arr).unsqueeze(0)
                # mask_arr = transforms.ToTensor()(mask_arr)
                img_arr = self.transforms(img_arr)
                return img_arr.float(), mask_arr.float()
            else: # cont
                img_arr0 = open_image(path_dict['img'][0]).astype(np.float64)
                img_arr1 = open_image(path_dict['img'][1]).astype(np.float64)
                mask_arr0 = open_image(path_dict['mask'][0])
                mask_arr1 = open_image(path_dict['mask'][1])
                loss = open_image(path_dict['loss'])
                mask_arr0 = torch.from_numpy(mask_arr0).unsqueeze(0)
                mask_arr1 = torch.from_numpy(mask_arr1).unsqueeze(0)

                # mask_arr = transforms.ToTensor()(mask_arr)
                img_arr0 = self.transforms(img_arr0)
                img_arr1 = self.transforms(img_arr1)

                loss = torch.from_numpy(loss).unsqueeze(0)
                return {
                    'imgs': (img_arr0.float(), img_arr1.float()),
                    'mask': (mask_arr0.float(), mask_arr1.float()),
                    'loss': loss.float()
                }
        else:
            img_arr0 = self.transforms(open_image(path_dict['img'][0]))
            img_arr1 = self.transforms(open_image(path_dict['img'][1]))
            img_arr2 = self.transforms(open_image(path_dict['img'][2]))
            img_arr3 = self.transforms(open_image(path_dict['img'][3]))
            loss = open_image(path_dict['loss'])
            mask_arr = open_image(path_dict['mask'])
            mask_arr = torch.from_numpy(mask_arr).unsqueeze(0)
            loss = torch.from_numpy(loss).unsqueeze(0)
            return {
                'imgs': (img_arr0.float(), img_arr1.float(),
                    img_arr2.float(), img_arr3.float()),
                'mask': mask_arr.float(),
                'loss': loss.float()
            }
