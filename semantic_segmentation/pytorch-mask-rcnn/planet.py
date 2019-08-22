"""Planet data adapted to the Coco format"""
import os
import glob
import cv2
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import datasets, transforms
import utils
from config import Config
from model import build_rpn_targets
from tqdm import tqdm
from skimage.io import imread
from skimage.segmentation import mark_boundaries
from skimage.measure import label, regionprops
from skimage.util import montage2d

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

# TODO: put get_imgs together.
def get_annual_imgs_from_mask(mask_file, img_dir):
    """
    Retrieve the annual input images from a mask.
    """
    year, z, x, y = get_tile_info(mask_file.split('/')[-1])
    key = str(year) + '_' + z + '_' + x + '_' + y
    planet_name = 'pl' + '{year}' + '_{z}_{x}_{y}.npy'
    planet_template = os.path.join(img_dir, planet_name)

    data = {}
    data[key] = {
        'img': (planet_template.format(year=year-1, z=z, x=x, y=y),
                planet_template.format(year=year, z=z, x=x, y=y)),
        'mask': mask_file
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
    year, z, x, y = get_tile_info(mask_file.split('/')[-1])
    planet_name = 'pl' + '{year}' + '_{q}_{z}_{x}_{y}.png'
    planet_template = os.path.join(img_dir, planet_name)

    data = {}
    if testing:
        key = str(year) + '_' + z + '_' + x + '_' + y
        # Put the same the quads together so it can be retrieved and plotted
        # at the same time while testing
        data[key] = {
            'img': (planet_template.format(year=year-1, q='q1', z=z, x=x, y=y),
                    planet_template.format(year=year-1, q='q2', z=z, x=x, y=y),
                    planet_template.format(year=year-1, q='q3', z=z, x=x, y=y),
                    planet_template.format(year=year-1, q='q4', z=z, x=x, y=y)),
            'mask': mask_file
        }
    else:
        key = str(year) + '_{q}_' + z + '_' + x + '_' + y
        quarters = ['q1_q2', 'q2_q3', 'q3_q4']
        for quarter in quarters:
            if quarter_type == 'same_year':
                img0 = planet_template.format(year=year-1, q=quarter[:2], z=z, x=x, y=y)
                img1 = planet_template.format(year=year-1, q=quarter[-2:], z=z, x=x, y=y)
                img = (img0, img1)
            else: # next_year
                img0 = planet_template.format(year=year-1, q=quarter[:2], z=z, x=x, y=y)
                img1 = planet_template.format(year=year, q=quarter[-2:], z=z, x=x, y=y)
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
        img_arr = np.load(img_path).transpose([1,2,0])
        return img_arr / 255.
    else:
        # For images transforms.ToTensor() does range to (0.,1.)
        img_arr = cv2.imread(img_path)
        return cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)

class BaseDataLoader(DataLoader):
    """
    Base class for all data loaders
    """
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle

        self.batch_idx = 0
        self.n_samples = len(dataset)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)
        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(sampler=self.sampler, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None

        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return DataLoader(sampler=self.valid_sampler, **self.init_kwargs)

class PlanetDataset(Dataset):
    """
    Planet 3-month mosaic dataset
    """
    def __init__(self, config, nn_config):
        """Initizalize dataset.
            Params:
                data_dir: absolute path, string
                years: list of years
                filetype: png or npy. If png it is raw data, if npy it has been preprocessed
        """
        self.config = config
        self.nn_config = nn_config
        self.data_dir = config['data_dir']
        self.years = config['years']
        self.qualities = config['qualities']
        self.timelapse = config['timelapse']
        self.max_dataset_size = config['max_dataset_size']
        self.training = config['training']
        self.testing = config['testing']
        self.quarter_type = config['quarter_type']

        if self.training:
            assert self.testing == False
        elif self.testing:
            assert self.training == False

        self.mask_dir = os.path.join(self.data_dir, 'hansen')
        self.img_dir = os.path.join(self.data_dir, 'planet')

        if self.timelapse == 'quarter':
            self.img_dir = os.path.join(self.img_dir, 'quarter')
            self.transforms = transforms.Compose([
                # transforms.RandomHorizontalFlip(), # try later, only works on imgs
                transforms.ToTensor(),
                Normalize((0.2397, 0.2852, 0.1837, 0.2397, 0.2852, 0.1837),
                    (0.1685, 0.1414, 0.1353, 0.1685, 0.1414, 0.1353))
            ])
        else: # annual
            self.img_dir = os.path.join(self.img_dir, 'annual')
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                Normalize((0.2311, 0.2838, 0.1752, 0.2311, 0.2838, 0.1752),
                    (0.1265, 0.0955, 0.0891, 0.1265, 0.0955, 0.0891))
            ])

        print('LOAD FILES INIT qualities', self.qualities)
        print('LOAD FILES INIT timelapse', self.timelapse)

        self.paths_dict = loadFiles(self.mask_dir, self.img_dir,
            self.years, self.qualities, self.timelapse, self.max_dataset_size,
            self.testing, self.quarter_type)

        self.keys = list(self.paths_dict.keys())
        self.dataset_size = len(self.paths_dict)

        # Anchors
        # [anchor_count, (y1, x1, y2, x2)]
        self.anchors = utils.generate_pyramid_anchors(nn_config.RPN_ANCHOR_SCALES,
                                                 nn_config.RPN_ANCHOR_RATIOS,
                                                 nn_config.BACKBONE_SHAPES,
                                                 nn_config.BACKBONE_STRIDES,
                                                 nn_config.RPN_ANCHOR_STRIDE)


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
            return self.__getsingleitem__(path_dict)
        else:
            return self.__getqitems__(path_dict)

    def __getqitems__(self, path_dict):
        # Testing mode quarterly. Plot images from the same quarter.
        img_arr0 = self.transforms(open_image(path_dict['img'][0]))
        img_arr1 = self.transforms(open_image(path_dict['img'][1]))
        img_arr2 = self.transforms(open_image(path_dict['img'][2]))
        img_arr3 = self.transforms(open_image(path_dict['img'][3]))
        img_arr_q1_q2 = torch.cat((img_arr0, img_arr1), 0)
        img_arr_q2_q3 = torch.cat((img_arr1, img_arr2), 0)
        img_arr_q3_q4 = torch.cat((img_arr2, img_arr3), 0)
        mask_arr = open_image(path_dict['mask'])
        mask_arr = transforms.ToTensor()(mask_arr[:,:,0])
        return {
            'imgs': (img_arr_q1_q2, img_arr_q2_q3, img_arr_q3_q4),
            'mask': mask_arr
        }

    def __getsingleitem__(self, path_dict):
        """Training mode or annual. Retrieve images as normal"""
        # image, image_metas, gt_class_ids, gt_boxes, gt_masks = \
        img_arr, img_meta, class_ids, bboxes, masks = \
            load_image_gt(path_dict, self.nn_config)

        # Take only R channel (labels) and transform to tensor. TODO: Check the other mask

        # img_arr0 = self.transforms(img_arr0)
        # img_arr1 = self.transforms(img_arr1)
        # img_arr = torch.cat((img_arr0, img_arr1), 0) # concatenate images
        # TODO: CHECK HOW build_rpn_targets work
        rpn_match, rpn_bbox = build_rpn_targets(img_arr.shape, self.anchors,
                                                class_ids, bboxes,
                                                self.nn_config)

        if bboxes.shape[0] > self.nn_config.MAX_GT_INSTANCES:
            ids = np.random.choice(
                np.arange(bboxes.shape[0]), self.nn_config.MAX_GT_INSTANCES, replace=False)
            class_ids = class_ids[ids]
            bboxes = bboxes[ids]
            masks = masks[:, :, ids]

        img_arr = self.transforms(img_arr)

        mask_arr = transforms.ToTensor()(masks).float()
        # Add to batch
        rpn_match = rpn_match[:, np.newaxis]
        # Convert
        img_meta = torch.from_numpy(img_meta)
        rpn_match = torch.from_numpy(rpn_match)
        rpn_bbox = torch.from_numpy(rpn_bbox).float()
        class_ids = torch.from_numpy(class_ids)
        bboxes = torch.from_numpy(bboxes).float()
        # return img_arr.float(), mask_arr.float()
        # print('img_arr', img_arr.size(), 'img_meta', img_meta.size(),
        # 'rpn_match', rpn_match.size(), 'rpn_bbox', rpn_bbox.size(),
        # 'class_ids', class_ids.size(), 'bboxes', bboxes.size(),
        # 'mask_arr', mask_arr.size())
        return img_arr.float(), img_meta, rpn_match, rpn_bbox, \
        class_ids, bboxes, mask_arr.float()

def load_image_gt(path_dict, config, augment=False, use_mini_mask=False):
    """
    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    img_arr0 = open_image(path_dict['img'][0])
    img_arr1 = open_image(path_dict['img'][1])
    mask_arr = open_image(path_dict['mask'])[:,:,0] # take only R channel

    bboxes = extract_bboxes(mask_arr)
    masks = []
    for bbox in bboxes:
        single_mask = get_single_mask(mask_arr, bbox)
        masks.append(single_mask)

    masks = np.stack(masks, axis=2)
    class_ids = np.ones([len(bboxes)], dtype=np.int32)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.ones([2], dtype=np.int32) # 2 classes, loss and bacground

    img_arr0, window, scale, padding = utils.resize_image(
        img_arr0,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)
    img_arr1, _, _, _ = utils.resize_image(
        img_arr1,
        min_dim=config.IMAGE_MIN_DIM,
        max_dim=config.IMAGE_MAX_DIM,
        padding=config.IMAGE_PADDING)
    masks = utils.resize_mask(masks, scale, padding)
    # window, scale and padding should be the same
    img_arr = np.concatenate((img_arr0, img_arr1), axis=2)
    img_meta = compose_image_meta(img_arr.shape,
        window, active_class_ids)
    return img_arr, img_meta, class_ids, bboxes, masks
    # return image, image_meta, class_ids, bbox, mask

def extract_bboxes(mask):
    """
    Extract bboxes from a mask according to label connected regions [1]
    [1]: https://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.label
    mask shape = [256, 256]
    """
    lbl = label(mask)
    props = regionprops(lbl)
    boxes = np.zeros([len(props), 4], dtype=np.int32)
    for i in range(len(props)):
        prop = props[i]
        boxes[i] = np.array(prop.bbox) # TODO: check if it is the correct order (y,x,y,x)
    return boxes # y0,x0,y1,x1

def get_single_mask(mask, bbox):
    """
    Get the mask that corresponds to a bounding box
    mask shape = [256, 256]
    """
    # box is y_upper_left,x_upper_left, y_lower_right, x_lower_right
    new_mask = np.zeros(mask.shape)
    upper_left_x, upper_left_y, lower_right_x, lower_right_y = bbox[1], bbox[0], bbox[3], bbox[2]
    # print(upper_left_y, lower_right_y, upper_left_x, lower_right_x)
    new_mask[upper_left_y:lower_right_y, upper_left_x:lower_right_x] = \
        mask[upper_left_y:lower_right_y, upper_left_x:lower_right_x]
    return new_mask

# def compose_image_meta(image_id, image_shape, window, active_class_ids):
def compose_image_meta(image_shape, window, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array. Use
    parse_image_meta() to parse the values back.

    image_id: An int ID of the image. Useful for debugging.
    image_shape: [height, width, channels]
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        list(image_shape) +     # size=3
        list(window) +          # size=4 (y1, x1, y2, x2) in image cooredinates
        list(active_class_ids)  # size=num_classes
    )
    return meta

class PlanetDataLoader(BaseDataLoader):

    def __init__(self, config, nn_config):

        # if training:
        #     subdir = os.path.join(data_dir, 'train')
        # else:
        #     subdir = os.path.join(data_dir, 'val')
        batch_size = config['batch_size']
        shuffle = config['shuffle']
        num_workers = config['num_workers']
        self.dataset = PlanetDataset(config, nn_config)
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

    dtype = tensor.type()
    # mean = torch.as_tensor(mean, dtype=dtype, device=tensor.device)
    # std = torch.as_tensor(std, dtype=dtype, device=tensor.device)
    # print('NORMALIZE DTYPE', dtype)
    mean = torch.DoubleTensor(mean)
    std = torch.DoubleTensor(std)
    # mean = mean.cuda()
    # std = std.cuda()
    tensor.sub_(mean[:, None, None]).div_(std[:, None, None])
    return tensor
