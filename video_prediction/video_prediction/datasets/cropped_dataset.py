import argparse
import glob
import itertools
import os
import pickle
import random
import re

import numpy as np
import skimage.io
import tensorflow as tf

from video_prediction.datasets.base_dataset import VarLenFeatureVideoDataset
from video_prediction.datasets.utils import get_list_of_files

class CroppedVideoDataset(VarLenFeatureVideoDataset):
    def __init__(self, *args, **kwargs):
        super(CroppedVideoDataset, self).__init__(*args, **kwargs)
        from google.protobuf.json_format import MessageToDict
        example = next(tf.python_io.tf_record_iterator(self.filenames[0]))
        dict_message = MessageToDict(tf.train.Example.FromString(example))
        feature = dict_message['features']['feature']
        image_shape = tuple(int(feature[key]['int64List']['value'][0]) for key in ['height', 'width', 'channels'])
        self.state_like_names_and_shapes['images'] = 'images/encoded', image_shape

    def get_default_hparams_dict(self):
        default_hparams = super(CroppedVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=2,
            sequence_length=4,
            # clip_length=1,
            # long_sequence_length=4,
            # force_time_shift=True,
            # shuffle_on_val=True,
            use_state=False,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    @property
    def jpeg_encoding(self):
        return False

    def num_examples_per_epoch(self):
        with open(os.path.join(self.input_dir, 'sequence_lengths.txt'), 'r') as sequence_lengths_file:
            sequence_lengths = sequence_lengths_file.readlines()
        sequence_lengths = [int(sequence_length.strip()) for sequence_length in sequence_lengths]
        return np.sum(np.array(sequence_lengths) >= self.hparams.sequence_length)


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# TODO: put in utils
def get_tile_info(tile):
    """
    Retrieve the year, zoom, x, y from a tile. Example: 2017_12_1223_2516.png
    """
    tile_name = tile.split('/')[-1]
    tile_items = tile_name.split('_')

    year = tile_items[0][2:]
    q = tile_items[1]
    z = tile_items[2]
    x = tile_items[3]
    y = tile_items[4]
    px = tile_items[5]
    py = tile_items[6][:-4]
    return year, q, z, x, y, px, py

def def_dic(img_dir, year, z, x, y, px, py):
    ntemp = os.path.join(img_dir, 'pl{year}_{q}_{z}_{x}_{y}_{px}_{py}.png')
    dic = {
            'q1': ntemp.format(year=year, q='q1', z=z, x=x, y=y, px=px, py=py),
            'q2': ntemp.format(year=year, q='q2', z=z, x=x, y=y, px=px, py=py),
            'q3': ntemp.format(year=year, q='q3', z=z, x=x, y=y, px=px, py=py),
            'q4': ntemp.format(year=year, q='q4', z=z, x=x, y=y, px=px, py=py)
    }
    return dic

def add_img(dic, img_dir, year, q, z, x, y, px, py):
    # There are images from 2016, 2017, 2018
    # Before it was designed for 2 images input 16-17 -> loss 17, 17-18 -> loss 18
    # It gets the images from 2016 and 2017
    # if 2016, 2017 is guaranteed
    # if 2017, 2016 is not guaranteed, 2018 is not guaranteed, but one of them is
    # if 2018, 2017 is guaranteed
    # In this case we dont do that (add both years, that was from not cropped. TODO update in the other script)
    key = None
    if year == '2016':
        key = '_'.join(('2016', z, x, y, px, py))
        year = '2016'
    elif year == '2017':
        key = '_'.join(('2017', z, x, y, px, py))
        year = '2017'
    elif year == '2018':
        key = '_'.join(('2018', z, x, y, px, py))
    if key:
        if key not in dic:
            dic[key] = def_dic(img_dir, int(year), z, x, y, px, py)

def get_imgs(img_dir, limit=float("inf")):
    data = {}
    img_paths = glob.glob(os.path.join(img_dir, '*.png')) # get the predictions of the generator (clean)
    for path in img_paths:
        year, q, z, x, y, px, py = get_tile_info(path)
        add_img(data, img_dir, year, q, z, x, y, px, py)
        if len(data) >= limit:
            break
    return data

def save_tf_record(output_fname, sequences):
    print('saving sequences to %s' % output_fname)
    with tf.python_io.TFRecordWriter(output_fname) as writer:
        for sequence in sequences:
            num_frames = len(sequence) # 6, 2 years
            height, width, channels = sequence[0].shape
            encoded_sequence = [image.tostring() for image in sequence]
            print('num_frames', num_frames, height, width, channels, len(encoded_sequence), sequence[0].dtype)
            features = tf.train.Features(feature={
                'sequence_length': _int64_feature(num_frames),
                'height': _int64_feature(height),
                'width': _int64_feature(width),
                'channels': _int64_feature(channels),
                'images/encoded': _bytes_list_feature(encoded_sequence),
            })
            example = tf.train.Example(features=features)
            writer.write(example.SerializeToString())

def get_quad_list(quad):
    return [
        quad['q1'],
        quad['q2'],
        quad['q3'],
        quad['q4'],
    ]

def read_frames_and_save_tf_records(output_dir, img_quads, image_size, partition_name, sequences_per_file=4):
    """
    img_quads: {
        key1: {year_q1: img1, year_q2: img2, year_q3: img3}
        key2: {year_q1: img1, year_q2: img2, year_q3: img3}
    }
    """
    partition_name = os.path.split(output_dir)[1]
    seq2img = {}
    sequences = []
    sequence_iter = 0
    sequence_lengths_file = open(os.path.join(output_dir, 'sequence_lengths.txt'), 'w')
    for video_iter, key in enumerate(img_quads.keys()):
        frame_fnames = get_quad_list(img_quads[key])
        frames = skimage.io.imread_collection(frame_fnames)
        frames = [frame[:,:,:3] for frame in frames] # take only RGB
        for f in frames:
            print(type(f), f.shape, f.meta)
        if not sequences:
            last_start_sequence_iter = sequence_iter
            print("reading sequences starting at sequence %d" % sequence_iter)
        sequences.append(frames)
        sequence_iter += 1
        sequence_lengths_file.write("%d\n" % len(frames)) # should be always 3
        if (len(sequences) == sequences_per_file or
                (video_iter == (len(img_quads) - 1))):
            output_fname = 'sequence_{0}_to_{1}.tfrecords'.format(last_start_sequence_iter, sequence_iter - 1)
            seq2img[key] = output_fname
            output_fname = os.path.join(output_dir, output_fname)
            print('SEQ2IMG', key, output_fname)
            save_tf_record(output_fname, sequences)
            sequences[:] = []
        if video_iter > 200:
            break
    sequence_lengths_file.close()

def part_dict(dic, num):
    total = len(dic)
    assert num < total
    rest = total - num
    dic1 = {}
    dic2 = {}
    itr = 0
    for key, value in dic.items():
        if itr < num:
            dic1[key] = value
        else:
            dic2[key] = value
        itr = itr + 1
    return dic1, dic2

# def partition_data(quads):
#     total_quads = len(quads)
#     num_train = int(0.8 * total_quads)
#     num_val = int(0.1 * total_quads)
#     print('===', num_train, num_val)
#     train_quads, test_quads = part_dict(quads, num_train)
#     train_quads, val_quads = part_dict(train_quads, num_train - num_val)
#     return [train_quads, val_quads, test_quads]

# def partition_data(quads)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, help="directory containing the quarter mosaics from planet")
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--image_size", type=int)
    args = parser.parse_args()

    partition_names = ['train', 'val', 'test']
    quad_list = [get_imgs(os.path.join(args.input_dir, 'train')),
        get_imgs(os.path.join(args.input_dir, 'val')),
        get_imgs(os.path.join(args.input_dir, 'test')),
    ]
    # quads = get_imgs(args.input_dir) # Return
#     train_quads, val_quads, test_quads
    # quad_list = partition_data(quads)
    print(len(quad_list[0]), len(quad_list[1]), len(quad_list[2]))
    for partition_name, partition_quad in zip(partition_names, quad_list):
    # for partition_name, partition_fnames in zip(partition_names, partition_fnames):
        partition_dir = os.path.join(args.output_dir, partition_name)
        if not os.path.exists(partition_dir):
            os.makedirs(partition_dir)
        read_frames_and_save_tf_records(partition_dir, partition_quad, args.image_size, partition_name)


if __name__ == '__main__':
    main()
