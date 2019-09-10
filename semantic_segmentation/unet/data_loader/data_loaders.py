"""
Data loader to test the outputs of the video prediction models
"""
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

LOSS_PATH_DB = '/mnt/ds3lab-scratch/lming/data/min_quality/forest_loss_yearly_cropped/test'
COVER_PATH_DB = '/mnt/ds3lab-scratch/lming/data/min_quality/forest_cover_cropped_processed/no_pct/test'

# with open('/mnt/ds3lab-scratch/lming/forest-prediction/video_prediction/test_imgs_gan.pkl', 'rb') as pkl_file:
#     TEST_IMGS = pkl.load(pkl_file)

# NOTE: INPUT_PATH_DB => input in the data loader thing

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir) if os.path.isdir(os.path.join(a_dir, name))]

TEST_IMGS = sorted(get_immediate_subdirectories('/mnt/ds3lab-scratch/lming/data/min_quality/planet/tfrecordsinfour/gan'))

def get_item(i, video_path):
    '''
    video_path = /mnt/ds3lab-scratch/lming/forest-prediction/video_prediction/results_today/gan/ours_deterministic_l1
    it has subfolders like TEST_IMGS, with the video images
    '''
    if video_path.split('/')[-2] == 'gan':
        gt_dir = '/mnt/ds3lab-scratch/lming/data/min_quality/planet/quarter_cropped_gan/test'
    else:
        gt_dir = '/mnt/ds3lab-scratch/lming/data/min_quality/planet/quarter_cropped/test'
    gt_template = os.path.join(gt_dir, 'pl{year}_{q}_{z}_{x}_{y}_{cx}_{cy}.png')
    key = TEST_IMGS[i]
    year, z, x, y, cx, cy = key.split('_')
    video_template = os.path.join(video_path, key, 'gen_image_{idx}_{latent}_{num_pred}.png')
    return {
        'gt_imgs': {'q1': gt_template.format(year=year, q='q1', z=z, x=x, y=y, cx=cx, cy=cy),
                    'q2': gt_template.format(year=year, q='q2', z=z, x=x, y=y, cx=cx, cy=cy),
                    'q3': gt_template.format(year=year, q='q3', z=z, x=x, y=y, cx=cx, cy=cy),
                    'q4': gt_template.format(year=year, q='q4', z=z, x=x, y=y, cx=cx, cy=cy)
        },
        'video_imgs': {'00': video_template.format(idx=str(0).zfill(5), latent='00', num_pred='00'),
                      '01': video_template.format(idx=str(0).zfill(5), latent='00', num_pred='01'),
                      '02': video_template.format(idx=str(0).zfill(5), latent='00', num_pred='02')},
        'forest_loss': os.path.join(LOSS_PATH_DB, 'ly{year}_{z}_{x}_{y}_{cx}_{cy}.npy'.format(
                year=year, z=z, x=x, y=y, cx=cx, cy=cy
        )),
        'forest_cover': os.path.join(COVER_PATH_DB, 'fc{year}_{z}_{x}_{y}_{cx}_{cy}.npy'.format(
                year=year, z=z, x=x, y=y, cx=cx, cy=cy
        ))
    }

# def get_item(i, video_path):
#     list_gan = list(TEST_IMGS)
#     key = list_gan[i]
#     year, z, x, y, cx, cy = key.split('_')
#     folder = get_folder128(i)
#     video_template = os.path.join(video_path, folder, 'gen_image_{idx}_{latent}_{num_pred}.png')
#     return {
#         'gt_imgs': TEST_IMGS[key],
#         'video_imgs': {
#                 '00': video_template.format(idx=str(i).zfill(5), latent='00', num_pred='00'),
#                 '01': video_template.format(idx=str(i).zfill(5), latent='00', num_pred='01'),
#                 '02': video_template.format(idx=str(i).zfill(5), latent='00', num_pred='02')
#         },
#         'forest_loss': os.path.join(LOSS_PATH_DB, 'ly{year}_{z}_{x}_{y}_{cx}_{cy}.npy'.format(
#                 year=year, z=z, x=x, y=y, cx=cx, cy=cy
#         )),
#         'forest_cover': os.path.join(COVER_PATH_DB, 'fc{year}_{z}_{x}_{y}_{cx}_{cy}.npy'.format(
#                 year=year, z=z, x=x, y=y, cx=cx, cy=cy
#         ))
#     }

# def get_folder128(idx):
#     init = (idx // 128) * 128
#     end = init + 128
#     return str(int(init)) + '_' + str(int(end))
#
# def loss2file(key, video_path):
#     gt_imgs = TEST_IMGS[key]
#     # gen_image_00030_00_00.png, idx always 5zeros, latent always 00 (unless you set more preds), 00, 01 is number of preds
#
#     idx_int = list(TEST_IMGS).index(key)
#     idx = str(idx_int).zfill(5)
#     folder = get_folder128(idx_int)
#
#     video_template = os.path.join(video_path, folder, 'gen_image_{idx}_{latent}_{num_pred}.png')
#     video_imgs = {
#         '00': video_template.format(idx=idx, latent='00', num_pred='00'),
#         '01': video_template.format(idx=idx, latent='00', num_pred='01'),
#         '02': video_template.format(idx=idx, latent='00', num_pred='02')
#     }
#     return gt_imgs, video_imgs
#

#
# def loadFiles(video_path, limit=float("inf")):
#     list_gan = list(TEST_IMGS)
#     imgs = {}
#     for i in range(128): # for some reason only works on the first path
#         key = list_gan[i]
#         year, z, x, y, cx, cy = key.split('_')
#         folder = get_folder128(i)
#         video_template = os.path.join(video_path, folder, 'gen_image_{idx}_{latent}_{num_pred}.png')
#         imgs[key] = {
#             'gt_imgs': TEST_IMGS[key],
#             'video_imgs': {
#                 '00': video_template.format(idx=str(i).zfill(5), latent='00', num_pred='00'),
#                 '01': video_template.format(idx=str(i).zfill(5), latent='00', num_pred='01'),
#                 '02': video_template.format(idx=str(i).zfill(5), latent='00', num_pred='02')
#             },
#             'forest_loss': os.path.join(LOSS_PATH_DB, 'ly{year}_{z}_{x}_{y}_{cx}_{cy}.npy'.format(
#                 year=year, z=z, x=x, y=y, cx=cx, cy=cy
#             )),
#             'forest_cover': os.path.join(COVER_PATH_DB, 'fc{year}_{z}_{x}_{y}_{cx}_{cy}.npy'.format(
#                 year=year, z=z, x=x, y=y, cx=cx, cy=cy
#             ))
#         }
#         if len(imgs) > limit:
#             imgs = {k: imgs[k] for k in list(imgs)[:limit]}
#             print('LOAD FILES', len(imgs))
#             return imgs
#     print('LOAD FILES', len(imgs))
#     return imgs
#
# def loadFiles1(video_path, limit=float("inf")):
#     imgs = {}
#     mask_imgs = glob.glob(os.path.join(LOSS_PATH_DB, '*'))
#     print('LOAD FILES MASK IMGS', len(mask_imgs))
#     for mask in mask_imgs:
#         year, z, x, y, cx, cy = get_tile_info(mask.split('/')[-1])
#         key = str(year) + '_' + z + '_' + x + '_' + y + '_' + cx + '_' + cy
#         gt_imgs, video_imgs = loss2file(key, video_path)
#         imgs[key] = {
#             'gt_imgs': gt_imgs,
#             'video_imgs': video_imgs,
#             'forest_loss': mask,
#             'forest_cover': os.path.join(COVER_PATH_DB, 'fc{year}_{z}_{x}_{y}_{cx}_{cy}.npy'.format(
#                 year=year, z=z, x=x, y=y, cx=cx, cy=cy
#             ))
#         }
#         if len(imgs) > limit:
#             imgs = {k: imgs[k] for k in list(imgs)[:limit]}
#             print('LOAD FILES', len(imgs))
#             return imgs
#     print('LOAD FILES', len(imgs))
#     return imgs

# TODO: put in utils
def get_tile_info(file):
    items = file.split('_')
    year, zoom, x, y, cx, cy = items[0][2:], items[1], items[2], items[3], items[4], items[5][:-4]
    return year, zoom, x, y, cx, cy


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
                # change to binary mask, sometimes we have [0 76]
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
        self.img_dir = input_dir
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            Normalize((0.2311, 0.2838, 0.1752),
                (0.1265, 0.0955, 0.0891))
        ])
        # if self.timelapse == 'quarter':
        #     self.img_dir = os.path.join(self.img_dir, 'quarter')
        #     self.transforms = transforms.Compose([
        #         # transforms.RandomHorizontalFlip(), # try later, only works on imgs
        #         transforms.ToTensor(),
        #         Normalize((0.2397, 0.2852, 0.1837),
        #             (0.1685, 0.1414, 0.1353))
        #     ])
        # else: # annual
        #     self.img_dir = os.path.join(self.img_dir, 'annual')
        #     self.transforms = transforms.Compose([
        #         transforms.ToTensor(),
        #         Normalize((0.2311, 0.2838, 0.1752),
        #             (0.1265, 0.0955, 0.0891))
        #     ])

        if max_dataset_size == 'inf':
            max_dataset_size = float('inf')
        # self.paths_dict = loadFiles(self.img_dir)
        # self.keys = list(self.paths_dict.keys())
        # self.dataset_size = len(self.paths_dict)
        self.dataset_size = len(TEST_IMGS)

    def __len__(self):
        # print('Planet Dataset len called')
        return self.dataset_size

    def __getitem__(self, index):
        r"""Returns data point and its binary mask"""
        # Notes: tiles in annual mosaics need to be divided by 255.
        #
        print('RETRIEVING', index)
        key = self.keys[index % self.dataset_size]
        # path_dict = self.paths_dict[key]
        path_dict = get_item(index, self.img_dir)
        # print('PATHHHH',path_dict['video_imgs']['00'])
        gt_img0 = self.transforms(open_image(path_dict['gt_imgs']['q1']))
        gt_img1 = self.transforms(open_image(path_dict['gt_imgs']['q2']))
        gt_img2 = self.transforms(open_image(path_dict['gt_imgs']['q3']))
        gt_img3 = self.transforms(open_image(path_dict['gt_imgs']['q4']))
        video_img0 = self.transforms(open_image(path_dict['video_imgs']['00']))
        video_img1 = self.transforms(open_image(path_dict['video_imgs']['01']))
        video_img2 = self.transforms(open_image(path_dict['video_imgs']['02']))
        forest_loss = open_image(path_dict['forest_loss'])
        forest_loss = torch.from_numpy(forest_loss).unsqueeze(0)
        forest_cover = open_image(path_dict['forest_cover'])
        forest_cover = torch.from_numpy(forest_cover).unsqueeze(0)

        return {
            'gt_imgs': {
                'q1': gt_img0,
                'q2': gt_img1,
                'q3': gt_img2,
                'q4': gt_img3,
            },
            'video_imgs':{
                '00': video_img0,
                '01': video_img1,
                '02': video_img2
            },
            'forest_loss': forest_loss,
            'forest_cover': forest_cover
        }

class PlanetDataLoader(BaseDataLoader):

    def __init__(self, input_dir,
            label_dir,
            batch_size,
            years,
            qualities,
            timelapse,
            max_dataset_size=float('inf'),
            shuffle=False,
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

def main():
    item = get_item(10, '/mnt/ds3lab-scratch/lming/forest-prediction/video_prediction/results_today/gan/ours_deterministic_l1')
    # files = glob.glob(os.path.join(LOSS_PATH_DB, '*'))
    # file0 = files[0]
    # year, z, x, y, cx, cy = get_tile_info(file0.split('/')[-1])
    # key = str(year) + '_' + z + '_' + x + '_' + y + '_' + cx + '_' + cy
    # gt_imgs, video_imgs = loss2file(key, '/mnt/ds3lab-scratch/lming/forest-prediction/video_prediction/results_today/planet_cropped_gan/ours_deterministic_l1')
    with open('test_images.pkl', 'wb') as pkl_file:
        pkl.dump(item, pkl_file)
if __name__ == '__main__':
    main()
