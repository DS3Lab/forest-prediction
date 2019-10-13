import json
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from torchvision.utils import make_grid
from torchvision import transforms
import os
from pathlib import Path
from datetime import datetime
from itertools import repeat
from collections import OrderedDict


def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_json(fname):
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_json(content, fname):
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader

def make_grid_2(tensor, nrow=8, padding=2, normalize=False, range=None, scale_each=False, pad_value=0):
    tensor1 = tensor[:,:3,:,:]
    tensor2 = tensor[:,3:,:,:]

    tensor = torch.cat([tensor1, tensor2], dim=3)
    return make_grid(tensor, nrow, padding, normalize, range, scale_each, pad_value)

def save_video_images256(images, out_dir, idx_start):
    num_y_tiles = 3
    f = plt.figure(figsize=(4, num_y_tiles*2))
    gs = gridspec.GridSpec(num_y_tiles, 5, wspace=0.0, hspace=0.0)
    years = ['2013', '2014', '2015', '2016', '2017']
    for i in range(0, len(years)):
        year = years[i]
        img = images[year]['img']
        gt = images[year]['gt']
        pred = images[year]['pred']
        ax = plt.subplot(gs[0, i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        print('plot 0')
        plt.imshow(np.transpose(img[0], axes=[1,2,0]))

        ax = plt.subplot(gs[1, i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        print('plot 1')
        print(gt.shape, 'SHAPEEE 1')
        plt.imshow(gt[0][0], cmap=plt.cm.binary)
        ax = plt.subplot(gs[2, i])
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        print('plot 2')
        plt.imshow(pred[0][0], cmap=plt.cm.binary)
    out_imgs_dir = os.path.join(out_dir, '{}.png'.format(idx_start))
    print('Saved!', out_imgs_dir)
    plt.savefig(out_imgs_dir, dpi=200, bbox_inches='tight', pad_inches=0.0)
    plt.close(f)

def save_simple_images(batch_size, images, out_dir, idx_start):

    for i in range(0, images['img'].shape[0], batch_size):
        num_y_tiles = 3 # input, gt, prediction
        f = plt.figure(figsize=(batch_size*4, num_y_tiles*2))
        gs = gridspec.GridSpec(num_y_tiles, batch_size, wspace=0.0, hspace=0.0)
        tiles = list(range(i, i + batch_size))

        for tile in tiles:
            # img1, img2, gt, pred
            if tile < images['img'].shape[0]:
                img = images['img'][tile]
                gt = images['gt'][tile]
                pred = images['pred'][tile]
                # Set up plot
                ax = plt.subplot(gs[0, tile%batch_size])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                print('plot 0')
                plt.imshow(np.transpose(img, axes=[1,2,0]))
                ax = plt.subplot(gs[1, tile%batch_size])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                print('plot 1')
                plt.imshow(gt[0], cmap=plt.cm.binary)
                ax = plt.subplot(gs[2, tile%batch_size])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                print('plot 2')
                plt.imshow(pred[0], cmap=plt.cm.binary)

        out_imgs_dir = os.path.join(out_dir, '{}.png'.format(i + idx_start))
        print('Saved!', out_imgs_dir)
        plt.savefig(out_imgs_dir, dpi=200, bbox_inches='tight', pad_inches=0.0)
        plt.close(f)

def save_double_images(batch_size, images, out_dir, idx_start):

    for i in range(0, images['img'].shape[0], batch_size):
        num_y_tiles = 4 # input, gt, prediction
        f = plt.figure(figsize=(batch_size*4, num_y_tiles*2))
        gs = gridspec.GridSpec(num_y_tiles, batch_size, wspace=0.0, hspace=0.0)
        tiles = list(range(i, i + batch_size))

        for tile in tiles:
            # img1, img2, gt, pred
            if tile < images['img'].shape[0]:
                img = images['img'][tile]
                gt = images['gt'][tile]
                pred = images['pred'][tile]
                # Set up plot
                ax = plt.subplot(gs[0, tile%batch_size])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                print('plot 0')
                plt.imshow(np.transpose(img[:3], axes=[1,2,0]))
                ax = plt.subplot(gs[1, tile%batch_size])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                print('plot 1')
                plt.imshow(np.transpose(img[3:], axes=[1,2,0]))
                ax = plt.subplot(gs[2, tile%batch_size])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                print('plot 2')
                plt.imshow(gt[0], cmap=plt.cm.binary)
                ax = plt.subplot(gs[3, tile%batch_size])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                print('plot 3')
                plt.imshow(pred[0], cmap=plt.cm.binary)

        out_imgs_dir = os.path.join(out_dir, '{}.png'.format(i + idx_start))
        print('Saved!', out_imgs_dir)
        plt.savefig(out_imgs_dir, dpi=200, bbox_inches='tight', pad_inches=0.0)
        plt.close(f)


def get_loss_pred(img0, img1):
    # img0 = quarter0
    # img1 = quarter1
    # quarter0 - quarter1 = loss 0->1
    mask = np.where(img1 == 1)
    img = np.copy(img0)
    img[mask] = 0
    return img

def save_images3m(batch_size, images, out_dir, idx_start):
    num_y_tiles = 2
    f = plt.figure(figsize=(batch_size*4, num_y_tiles*2))
    gs = gridspec.GridSpec(3, 1, wspace=0.0, hspace=0.0)
    mask_recreation_gt = images['mask_recreation_gt']
    mask_recreation_pred = images['mask_recreation_pred']

    ax = plt.subplot(gs[0, 0])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    print('plot 0')
    plt.imshow(mask_recreation_gt, cmap=plt.cm.binary)

    ax = plt.subplot(gs[1, 0])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    print('plot 1')
    plt.imshow(mask_recreation_pred, cmap=plt.cm.binary)

    out_imgs_dir = os.path.join(out_dir, '{}.png'.format(idx_start))
    print('Saved!', out_imgs_dir)
    plt.savefig(out_imgs_dir, dpi=200, bbox_inches='tight', pad_inches=0.0)
    plt.close(f)

def save_video_images(batch_size, images, out_dir, idx_start, limit):

    for i in range(0, images['video_imgs'].shape[0], batch_size):
        num_y_tiles = 5 # video_img, forest cover, prediction cover, forest loss, prediction loss
        print(images['video_imgs'].shape, 'SAVE VIDEO IMAGES CHECK')
        f = plt.figure(figsize=(batch_size*4, num_y_tiles*2))
        gs = gridspec.GridSpec(num_y_tiles, batch_size, wspace=0.0, hspace=0.0)
        tiles = list(range(i, i + batch_size))

        for tile in tiles:
            # img1, img2, gt, pred
            video_img = images['video_imgs'][tile]
            forest_cover = images['forest_cover'][tile]
            prediction_cover = images['prediction_cover'][tile]
            forest_loss = images['forest_loss'][tile]
            if tile > i:
                prediction_loss = get_loss_pred(images['prediction_cover'][tile-1],
                                     images['prediction_cover'][tile])
            else:
                prediction_loss = np.zeros((prediction_cover.shape))
            # Set up plot
            ax = plt.subplot(gs[0, tile%batch_size])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print('plot 0')
            plt.imshow(np.transpose(video_img, axes=[1,2,0]))
            ax = plt.subplot(gs[1, tile%batch_size])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print('plot 1')
            plt.imshow(forest_cover[0], cmap=plt.cm.binary)
            ax = plt.subplot(gs[2, tile%batch_size])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print('plot 2')
            plt.imshow(prediction_cover[0], cmap=plt.cm.binary)

            ax = plt.subplot(gs[3, tile%batch_size])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print('plot 3')
            plt.imshow(forest_loss[0], cmap=plt.cm.binary)

            ax = plt.subplot(gs[4, tile%batch_size])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print('plot 4')
            plt.imshow(prediction_loss[0], cmap=plt.cm.binary)
        out_imgs_dir = os.path.join(out_dir, '{}.png'.format(i + idx_start))
        print('Saved!', out_imgs_dir)
        plt.savefig(out_imgs_dir, dpi=200, bbox_inches='tight', pad_inches=0.0)
        plt.close(f)

def save_images_single(batch_size, images, out_dir, idx_start, limit):

    for i in range(0, images['img'].shape[0], batch_size):
        num_y_tiles = 5
        f = plt.figure(figsize=(batch_size*4, num_y_tiles*2))
        gs = gridspec.GridSpec(num_y_tiles, batch_size, wspace=0.0, hspace=0.0)
        tiles = list(range(i, i + batch_size))

        for tile in tiles:
            # img1, img2, gt, pred
            img = images['img'][tile]
            gt = images['gt'][tile]
            pred = images['pred'][tile]
            loss_gt = images['loss'][tile]
            if tile > i:
                loss = get_loss_pred(images['pred'][tile-1], images['pred'][tile-2])
            else:
                loss = np.zeros((pred.shape))
            # Set up plot
            ax = plt.subplot(gs[0, tile%batch_size])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print('plot 0')
            plt.imshow(np.transpose(img, axes=[1,2,0]))
            ax = plt.subplot(gs[1, tile%batch_size])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print('plot 1')
            plt.imshow(gt[0], cmap=plt.cm.binary)
            ax = plt.subplot(gs[2, tile%batch_size])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print('plot 2')
            plt.imshow(pred[0], cmap=plt.cm.binary)

            ax = plt.subplot(gs[3, tile%batch_size])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print('plot 3')
            plt.imshow(loss[0], cmap=plt.cm.binary)

            ax = plt.subplot(gs[4, tile%batch_size])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print('plot 4')
            plt.imshow(loss_gt[0], cmap=plt.cm.binary)
        out_imgs_dir = os.path.join(out_dir, '{}.png'.format(i + idx_start))
        print('Saved!', out_imgs_dir)
        plt.savefig(out_imgs_dir, dpi=200, bbox_inches='tight', pad_inches=0.0)
        plt.close(f)


def save_images_double(batch_size, images, out_dir, idx_start, limit):
    for i in range(0, images['img'].shape[0], batch_size):
        num_y_tiles = 4
        f = plt.figure(figsize=(batch_size*4, num_y_tiles*2))
        gs = gridspec.GridSpec(num_y_tiles, batch_size, wspace=0.0, hspace=0.0)
        tiles = list(range(i, i + batch_size))

        for tile in tiles:
            # img1, img2, gt, pred
            img = images['img'][tile]
            gt = images['gt'][tile]
            pred = images['pred'][tile]
            # Set up plot
            ax = plt.subplot(gs[0, tile%batch_size])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print('plot 0')
            plt.imshow(np.transpose(img, axes=[1,2,0])[:,:,:3])
            ax = plt.subplot(gs[1, tile%batch_size])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print('plot 1')
            plt.imshow(np.transpose(img, axes=[1,2,0])[:,:,3:6])
            ax = plt.subplot(gs[2, tile%batch_size])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print('plot 2')
            plt.imshow(gt[0], cmap=plt.cm.binary)
            ax = plt.subplot(gs[3, tile%batch_size])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print('plot 3')
            plt.imshow(pred[0], cmap=plt.cm.binary)
        out_imgs_dir = os.path.join(out_dir, '{}.png'.format(i + idx_start))
        print('Saved!', out_imgs_dir)
        plt.savefig(out_imgs_dir, dpi=200, bbox_inches='tight', pad_inches=0.0)
        plt.close(f)

def save_images(batch_size, images, out_dir, idx_start, input_type, limit=None):
#     out_dir = os.path.join(experiment, 'eval_year_images')
    """
    :param batch_size - num of predictions in the grid
    """
    # print('SAVE IMAGES LENGTH', len(images['img']), images['img'].shape)
    # np.save(os.path.join(out_dir, 'pred' + str(idx_start) + '.npy'), images['pred'])
    # np.save(os.path.join(out_dir, 'img' + str(idx_start) + '.npy'), images['img'])
    # np.save(os.path.join(out_dir, 'gt' + str(idx_start) + '.npy'), images['gt'])
    if input_type == 'one':
        save_images_single(batch_size, images, out_dir, idx_start, limit)
    else:
        save_images_double(batch_size, images, out_dir, idx_start, limit)
    """
    for i in range(0, images['img'].shape[0], batch_size):
        num_y_tiles = 4
        f = plt.figure(figsize=(batch_size*4, num_y_tiles*2))
        gs = gridspec.GridSpec(num_y_tiles, batch_size, wspace=0.0, hspace=0.0)
        tiles = list(range(i, i + batch_size))

        for tile in tiles:
            # img1, img2, gt, pred
            img = images['img'][tile]
            gt = images['gt'][tile]
            pred = images['pred'][tile]
            # Set up plot
            ax = plt.subplot(gs[0, tile%batch_size])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print('plot 0')
            plt.imshow(np.transpose(img, axes=[1,2,0])[:,:,:3])
            ax = plt.subplot(gs[1, tile%batch_size])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print('plot 1')
            plt.imshow(np.transpose(img, axes=[1,2,0])[:,:,3:6])
            ax = plt.subplot(gs[2, tile%batch_size])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print('plot 2')
            plt.imshow(gt[0], cmap=plt.cm.binary)
            ax = plt.subplot(gs[3, tile%batch_size])
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            print('plot 3')
            plt.imshow(pred[0], cmap=plt.cm.binary)
        out_imgs_dir = os.path.join(out_dir, '{}.png'.format(i + idx_start))
        print('Saved!', out_imgs_dir)
        plt.savefig(out_imgs_dir, dpi=200, bbox_inches='tight', pad_inches=0.0)
        plt.close(f)
    """

class Timer:
    def __init__(self):
        self.cache = datetime.now()

    def check(self):
        now = datetime.now()
        duration = now - self.cache
        self.cache = now
        return duration.total_seconds()

    def reset(self):
        self.cache = datetime.now()

class NormalizeInverse(transforms.Normalize):
    """
    Undoes the normalization and returns the reconstructed images in the input domain [1].
    [1] https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/3
    """

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())
