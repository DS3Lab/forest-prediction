import argparse
import glob
import itertools
import os
import pickle as pkl
import random
import re

import numpy as np
import skimage.io
import cv2
import tensorflow as tf
import imageio
def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _bytes_list_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=values))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _float_list_feature(values):
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))

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
                'images/encoded': _float_list_feature(sequence),
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

def open_and_whiten(img_paths, train_mean, white_matrix):
    X = []
    for path in img_paths:
        img_arr = cv2.imread(path)
        img_arr = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
        X.append(img_arr)
    X = np.array(X)
    N, H, W, C = X.shape
    X = X.reshape(X.shape[0], -1)
    # print('Data reshape,', X.shape)

    # print('Normalizing data...')
    X_norm = X / 255
    X_norm = X_norm - train_mean
    X_white = np.dot(X_norm, white_matrix.T)
    X_white = X_white.reshape(X_white.shape[0], H, W, C)
    # print('Data shape', X_white[0].shape)
    return [X_white[i, :, : ,:] for i in range(N)]


def read_frames_and_save_tf_records(output_dir, img_quads, image_size, white_params, sequences_per_file=128):
    """
    img_quads: {
        key1: {year_q1: img1, year_q2: img2, year_q3: img3}
        key2: {year_q1: img1, year_q2: img2, year_q3: img3}
    }
    """
    train_mean = white_params['mean']
    white_matrix = white_params['white_matrix']
    partition_name = os.path.split(output_dir)[1]

    sequences = []
    sequence_iter = 0
    sequence_lengths_file = open(os.path.join(output_dir, 'sequence_lengths.txt'), 'w')
    for video_iter, key in enumerate(img_quads.keys()):
        frame_fnames = get_quad_list(img_quads[key])
        # frame_fnames = [quads['q1'], quads['q2'], quads['q3'], quads['q4']]
        # frames = skimage.io.imread_collection(frame_fnames)
        # frames = [frame[:,:,:3] for frame in frames] # take only RGB
        # frames_raw = [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in frame_fnames]
        frames = open_and_whiten(frame_fnames, train_mean, white_matrix)
        frames = [imageio.core.util.Array(frame) for frame in frames]
        # for f in frames:
        #     print(f.shape)
        # save = {
        #     'raw': frames_raw,
        #     'white': frames
        # }
        # with open('compare_frames.pkl', 'wb') as pkl_file:
        #     pkl.dump(save, pkl_file)
        # break
        if not sequences:
            last_start_sequence_iter = sequence_iter
            print("reading sequences starting at sequence %d" % sequence_iter)
        sequences.append(frames)
        sequence_iter += 1
        sequence_lengths_file.write("%d\n" % len(frames)) # should be always 3
        if (len(sequences) == sequences_per_file or
                (video_iter == (len(img_quads) - 1))):
            output_fname = 'sequence_{0}_to_{1}.tfrecords'.format(last_start_sequence_iter, sequence_iter - 1)
            output_fname = os.path.join(output_dir, output_fname)
            save_tf_record(output_fname, sequences)
            sequences[:] = []
        
        if video_iter == 500:
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
    with open('white_matrix.pkl', 'rb') as pkl_file:
        white_matrix = pkl.load(pkl_file)
    with open('train_mean_std.pkl', 'rb') as pkl_file:
        mean_std = pkl.load(pkl_file)
    white_params = {
        'mean': mean_std['mean'],
        'white_matrix': white_matrix
    }
    for partition_name, partition_quad in zip(partition_names, quad_list):
    # for partition_name, partition_fnames in zip(partition_names, partition_fnames):
        partition_dir = os.path.join(args.output_dir, partition_name)
        if not os.path.exists(partition_dir):
            os.makedirs(partition_dir)
        read_frames_and_save_tf_records(partition_dir, partition_quad, args.image_size, white_params)


if __name__ == '__main__':
    main()
