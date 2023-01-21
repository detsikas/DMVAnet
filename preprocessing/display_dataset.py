import tensorflow as tf
import argparse
import os
import sys
from functools import partial
import preprocessing_dataset_utils as ds_utils
from common.image_utils import show_images
import numpy as np


parser = argparse.ArgumentParser(description='Display the images of a tf record')
parser.add_argument('--tfrecord', required=True, help='tfrecord file')
parser.add_argument('--correct-colors', help='RGB channel order', action='store_true')
parser.add_argument('--hide-y', help='Do not show the Y image', action='store_true')
args = parser.parse_args()

tfrecord = args.tfrecord
correct_colors = args.correct_colors
hide_y = args.hide_y

if not os.path.exists(tfrecord):
    print('Bad tf record path')
    sys.exit(0)

dataset = tf.data.TFRecordDataset([tfrecord])

parsed_dataset = dataset.map(partial(ds_utils.read_tf_record, with_filenames=True))
count = 0
for X_sample, Y_sample, filename in parsed_dataset:
    print('Displaying {}'.format(filename.numpy().decode()), end='\r')
    string_filename = filename.numpy().decode()

    images_to_show = []
    titles = ['X']
    grid = (1, 1)
    if correct_colors:
        images_to_show.append((np.flip(X_sample.numpy(), axis=-1) * 255).astype('uint8'))
    else:
        images_to_show.append((X_sample.numpy() * 255).astype('uint8'))

    if not hide_y:
        grid = (1, 2)
        titles.append('Y')
        Y_gray = (Y_sample.numpy() * 255).astype('uint8')
        if len(Y_gray.shape) == 2:
            Y_rgb = np.stack((Y_gray, Y_gray, Y_gray))
            Y_rgb = np.moveaxis(Y_rgb, 0, -1)
        else:
            Y_rgb = Y_gray
        images_to_show.append(Y_rgb)

    show_images(images_to_show, titles, grid)

print('\n Done')
