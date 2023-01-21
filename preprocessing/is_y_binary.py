import tensorflow as tf
import argparse
import os
import sys
from functools import partial
import preprocessing_dataset_utils as ds_utils
import numpy as np


parser = argparse.ArgumentParser(description='Display the images of a tf record')
input_group = parser.add_mutually_exclusive_group(required=True)
input_group.add_argument('--tfrecord', help='tfrecord file')
input_group.add_argument('--tfrecord-dir', help='tfrecord directory')
parser.add_argument('--show-images', help='Show non binary images', action='store_true')
args = parser.parse_args()

tfrecord = args.tfrecord
tfrecord_dir = args.tfrecord_dir
show_images = args.show_images


def check_tfrecord(input_tfrecord):
    dataset = tf.data.TFRecordDataset([input_tfrecord])
    parsed_dataset = dataset.map(partial(ds_utils.read_tf_record, with_filenames=True))
    count = 0
    for X_sample, Y_sample, filename in parsed_dataset:
        count += 1
        print('Checking {}'.format(filename.numpy().decode()), end='\r')
        Y = Y_sample.numpy()
        try:
            assert(len(Y.shape) == 2)
            assert((np.count_nonzero(Y == 0)+np.count_nonzero(Y == 1)) == (256*256))
        except AssertionError:
            print()
            print('Error in: {}'.format(filename))
            print(Y.shape)
            print(Y.shape[0]*Y.shape[1])
            print(np.count_nonzero(Y == 0))
            print(np.count_nonzero(Y == 1))
            Y[Y <= 0.7] = 0
            Y[Y > 0.7] = 1
            print(256*256-np.count_nonzero(Y == 1)-np.count_nonzero(Y == 0))
            if show_images:
                images_to_show = []
                titles = ['X']
                images_to_show.append((X_sample.numpy() * 255).astype('uint8'))
                grid = (1, 2)
                titles.append('Y')
                Y_gray = (Y * 255).astype('uint8')
                Y_rgb = np.stack((Y_gray, Y_gray, Y_gray))
                Y_rgb = np.moveaxis(Y_rgb, 0, -1)
                images_to_show.append(Y_rgb)

                show_images(images_to_show, titles, grid)
            exit(1)

    print()
    print('Checked {} patches'.format(count))


if tfrecord is not None:
    if not os.path.exists(tfrecord):
        print('Bad tf record path')
        sys.exit(0)

    check_tfrecord(tfrecord)
elif tfrecord_dir is not None:
    if not os.path.exists(tfrecord_dir):
        print('Bad tf record path')
        sys.exit(0)

    for root, dirs, files in os.walk(tfrecord_dir):
        for f in files:
            basename, ext = os.path.splitext(f)
            if ext == '.tfrecord':
                tfrecord_file = os.path.join(root, f)
                print('Reading from: {}'.format(tfrecord_file))
                check_tfrecord(tfrecord_file)

