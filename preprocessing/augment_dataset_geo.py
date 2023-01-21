import argparse
import tensorflow as tf
import os
import sys
from functools import partial
import preprocessing_dataset_utils as ds_utils
import numpy as np
import matplotlib.pyplot as plt


def show_image(title, img):
    fig = plt.figure()
    fig.suptitle(title)
    plt.imshow(img, cmap='gray')


parser = argparse.ArgumentParser(description='Apply geometrical augmentation to the dataset')
parser.add_argument('--tfrecord-file', required=True, help='Path to tf record file')
parser.add_argument('--rotation-frequency', type=float, default=0.3, help='How often to rotate a patch')
parser.add_argument('--flip-frequency', type=float, default=0.3, help='How often to flip a patch')
parser.add_argument('--output-path', help='Output path', required=True)
parser.add_argument('--debug', action='store_true', help='Show images')
args = parser.parse_args()

tfrecord_file = args.tfrecord_file
rotation_frequency = args.rotation_frequency
flip_frequency = args.flip_frequency
output_path = args.output_path
debug = args.debug

if not os.path.exists(tfrecord_file) or not os.path.isfile(tfrecord_file):
    print('Bad tf record file')

if not os.path.exists(output_path) or not os.path.isdir(output_path):
    os.makedirs(output_path)

basename = os.path.basename(tfrecord_file)
stem, ext = os.path.splitext(basename)
output_dataset_file = os.path.join(output_path, '{}_augment_geo{}'.format(stem, ext))

dataset = tf.data.TFRecordDataset([tfrecord_file])
feature_description = {
    'X': tf.io.FixedLenFeature([], tf.string),
    'Y': tf.io.FixedLenFeature([], tf.string),
    'filename': tf.io.FixedLenFeature([], tf.string)
}


def _read_tf_record(example_proto, with_filenames=False):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    raw_sample = tf.io.parse_single_example(example_proto, feature_description)
    X = tf.io.parse_tensor(raw_sample['X'], out_type=tf.float32)
    Y = tf.io.parse_tensor(raw_sample['Y'], out_type=tf.float32)
    if with_filenames:
        filename = raw_sample['filename']
        return X, Y, filename

    return X, Y


parsed_dataset = dataset.map(partial(_read_tf_record, with_filenames=True))

output_dataset_writer = tf.io.TFRecordWriter(output_dataset_file)

iterator = iter(parsed_dataset)
count = 0
rotation_count = 0
flip_count = 0
for X_sample, Y_sample, filename in iterator:
    print('Augmenting {}', filename, end='\r')
    if debug:
        print('\n')
    count += 1
    rotation_coin_toss_success = np.random.rand() <= rotation_frequency
    flip_coin_toss_success = np.random.rand() <= flip_frequency
    X_image = X_sample.numpy()
    Y_image = Y_sample.numpy()

    X_image_augmented = np.copy(X_image)
    Y_image_augmented = np.copy(Y_image)

    if rotation_coin_toss_success:
        rotation_count += 1
        number_of_rotations = np.random.randint(1, 4)
        if debug:
            print('Rotating by: {}'.format(number_of_rotations*90))
        for i in range(number_of_rotations):
            X_image_augmented = np.rot90(X_image_augmented)
            Y_image_augmented = np.rot90(Y_image_augmented)

    if flip_coin_toss_success:
        flip_count += 1
        flip_axis = 0 if np.random.rand() < 0.5 else 1
        if debug:
            print('Flipping along axis {}'.format(flip_axis))
        X_image_augmented = np.flip(X_image_augmented, axis=flip_axis)
        Y_image_augmented = np.flip(Y_image_augmented, axis=flip_axis)

    if debug:
        show_image('Original image', X_image)
        show_image('Original GT image', Y_image)
        if rotation_coin_toss_success or flip_coin_toss_success:
            show_image('Rotated image', X_image_augmented)
            show_image('Rotated GT_image', Y_image_augmented)
        plt.show()

    if rotation_coin_toss_success or flip_coin_toss_success:
        ds_utils.np_to_tf_record(X_image_augmented, Y_image_augmented, filename.numpy(), output_dataset_writer)

print()
print('Total dataset elements: {}'.format(count))
print('Flipped elements: {}'.format(flip_count))
print('Rotated elements: {}'.format(rotation_count))
print('Requested/actual flip frequency: {}/{}'.format(flip_frequency, flip_count/count))
print('Requested/actual rotation frequency: {}/{}'.format(rotation_frequency, rotation_count/count))

