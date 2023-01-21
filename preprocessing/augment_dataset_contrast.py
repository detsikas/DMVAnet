import argparse
import tensorflow as tf
import os
from functools import partial
import preprocessing_dataset_utils as ds_utils
import numpy as np
import matplotlib.pyplot as plt
import common.image_utils as image_utils


def show_image(title, img):
    fig = plt.figure()
    fig.suptitle(title)
    plt.imshow(img, cmap='gray')


parser = argparse.ArgumentParser(description='Augment the dataset by reducing the image contrast')
parser.add_argument('--tfrecord-file', required=True, help='Path to tf record file')
parser.add_argument('--augmentation-frequency', type=float, default=0.3, help='How often to augmkent a patch')
parser.add_argument('--alpha', help="ax+b", type=float)
parser.add_argument('--beta', help="ax+b", type=float)
parser.add_argument('--output-path', help='Output path', required=True)
parser.add_argument('--debug', action='store_true', help='Show images')

args = parser.parse_args()

tfrecord_file = args.tfrecord_file
augmentation_frequency = args.augmentation_frequency
alpha = args.alpha
beta = args.beta
output_path = args.output_path
debug = args.debug

if not os.path.exists(tfrecord_file) or not os.path.isfile(tfrecord_file):
    print('Bad tf record file')

if not os.path.exists(output_path) or not os.path.isdir(output_path):
    os.makedirs(output_path)

basename = os.path.basename(tfrecord_file)
stem, ext = os.path.splitext(basename)
output_dataset_file = os.path.join(output_path, '{}_augment_contrast{}'.format(stem, ext))

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
augmentation_count = 0
flip_count = 0

print('Contrast Augmentation by: {} {}'.format(alpha, beta))

for X_sample, Y_sample, filename in iterator:
    print('Augmenting {}', filename, end='\r')
    if debug:
        print('\n')
    count += 1
    augmentation_coin_toss_success = np.random.rand() <= augmentation_frequency
    X_image = X_sample.numpy()
    Y_image = Y_sample.numpy()

    if augmentation_coin_toss_success:
        augmentation_count += 1
        augmentation_level = 0
        X_image_augmented = image_utils.reduce_contrast(X_image*255, alpha, beta)/255
        if debug:
            show_image('Original image', X_image)
            show_image('Original GT image', Y_image)
            if augmentation_coin_toss_success:
                show_image('Augmented image', (255*X_image_augmented).astype('uint8'))
            plt.show()
        ds_utils.np_to_tf_record(X_image_augmented, Y_image, filename.numpy(), output_dataset_writer)

print()
print('Total dataset elements: {}'.format(count))
print('Flipped elements: {}'.format(flip_count))
print('Rotated elements: {}'.format(augmentation_count))
print('Requested/actual augmentation frequency: {}/{}'.format(augmentation_frequency, augmentation_count/count))


