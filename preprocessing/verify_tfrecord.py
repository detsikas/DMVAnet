import tensorflow as tf
import argparse
import os
import sys
from functools import partial
import numpy as np
import cv2
import preprocessing_dataset_utils as ds_utils

parser = argparse.ArgumentParser(description='Verify the contents of a tf record')
parser.add_argument('--tfrecord', required=True, help='tfrecord file')
parser.add_argument('--x-images-root-path', required=True, help='Path to the x images contained in the tfrecord')
parser.add_argument('--y-images-root-path', required=True, help='Path to the y images contained in the tfrecord')
parser.add_argument('--display-images', action='store_true')
args = parser.parse_args()

tfrecord = args.tfrecord
x_images_root_path = args.x_images_root_path
y_images_root_path = args.y_images_root_path
display_images = args.display_images

if not os.path.exists(tfrecord):
    print('Bad tf record path')
    sys.exit(0)

if not os.path.exists(x_images_root_path) or not os.path.isdir(x_images_root_path):
    print('Bad x images root path')
    sys.exit(0)

if not os.path.exists(y_images_root_path) or not os.path.isdir(y_images_root_path):
    print('Bad y images root path')
    sys.exit(0)

dataset = tf.data.TFRecordDataset([tfrecord])

parsed_dataset = dataset.map(partial(ds_utils.read_tf_record, with_filenames=True))
mean_abs_diff = 0
max_abs_diff = -1
sum_abs_diff = 0
count = 0
filenames = set()
num_training_files = len([f for f in os.listdir(x_images_root_path) if os.path.isfile(os.path.join(x_images_root_path, f))])
num_gt_files = len([f for f in os.listdir(y_images_root_path) if os.path.isfile(os.path.join(y_images_root_path, f))])

if num_gt_files != num_training_files:
    print('Mismatch in counts of x and y images')
    sys.exit(0)

for X_sample, Y_sample, filename in parsed_dataset:
    print('Verifying {}'.format(filename.numpy().decode()), end='\r')
    string_filename = filename.numpy().decode()
    if string_filename in filenames:
        print('\rDuplicate filename in dataset: {}'.format(filename))
        sys.exit()
    filenames.update([string_filename])

    if Y_sample.ndim != 2:
        print('\rBad Y dimensions: {} - {}'. format(filename, Y_sample.shape))
        sys.exit()

    Y_sample = np.stack((Y_sample,)*3, axis=-1)

    full_x_file_path = os.path.join(x_images_root_path, string_filename)
    full_y_file_path = os.path.join(y_images_root_path, string_filename)
    ref_x = np.load(full_x_file_path)
    ref_y = np.load(full_y_file_path)
    diff = np.abs(np.mean(X_sample - ref_x) + np.mean(Y_sample - ref_y))
    sum_abs_diff += diff
    if diff > max_abs_diff:
        max_abs_diff = diff
    count += 1
    if diff >= 1e-3:
        print('\nError in: {}!'.format(filename))
        sys.exit(0)
    if display_images:
        cv2.imshow('X', (X_sample.numpy() * 255).astype('uint8'))
        cv2.imshow('Y', (Y_sample * 255).astype('uint8'))
        cv2.imshow('refx', (ref_x * 255).astype('uint8'))
        cv2.imshow('refy', (ref_y * 255).astype('uint8'))
        cv2.waitKey(0)

if count != num_training_files:
    print('Mismatch in counts of dataset elements and x/y image files')
    sys.exit(0)

mean_abs_diff = sum_abs_diff / count
print()
print('No error')
print('Mean diff: {}, max diff: {}, sum diff: {}'.format(mean_abs_diff, max_abs_diff, sum_abs_diff))
