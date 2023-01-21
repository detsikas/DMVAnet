import argparse
import tensorflow as tf
import os
from functools import partial
import preprocessing_dataset_utils as ds_utils
import numpy as np
import matplotlib.pyplot as plt
from common.noise_augmentation import gaussian_noise, sp_noise
from common.image_utils import show_images, scale_image
import sys


parser = argparse.ArgumentParser(description='Augment the dataset by upscaling the images')
parser.add_argument('--tfrecord-file', required=True, help='Path to tf record file')
parser.add_argument('--augmentation-frequency', type=float, default=0.3,
                    help='How often to scale')
parser.add_argument('--scale', help='Image scaling to apply', type=float, default=2)
parser.add_argument('--output-path', help='Output path', required=True)
parser.add_argument('--debug', action='store_true', help='Show images')
args = parser.parse_args()

tfrecord_file = args.tfrecord_file
augmentation_frequency = args.augmentation_frequency
scale = args.scale
output_path = args.output_path
debug = args.debug

if scale <= 1:
    print('Upscale only')
    sys.exit(0)

if not os.path.exists(tfrecord_file) or not os.path.isfile(tfrecord_file):
    print('Bad tf record file')

if not os.path.exists(output_path) or not os.path.isdir(output_path):
    os.makedirs(output_path)

basename = os.path.basename(tfrecord_file)
stem, ext = os.path.splitext(basename)
output_dataset_file = os.path.join(output_path, '{}_augment_scale{}'.format(stem, ext))

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
        file_name = raw_sample['filename']
        return X, Y, file_name

    return X, Y


parsed_dataset = dataset.map(partial(_read_tf_record, with_filenames=True))

output_dataset_writer = tf.io.TFRecordWriter(output_dataset_file)

iterator = iter(parsed_dataset)
count = 0
augmentation_count = 0

print('Scale Augmentation by: {}'.format(scale))

for X_sample, Y_sample, filename in iterator:
    count += 1

    augmentation_coin_toss_success = np.random.rand() <= augmentation_frequency

    X_image = X_sample.numpy()
    Y_image = Y_sample.numpy()

    if augmentation_coin_toss_success:

        print('Augmenting {}', filename, end='\r')
        if debug:
            print('\n')
        augmentation_count += 1
        augmentation_level = 0

        X_image_augmented = (scale_image(255*X_image, scale)/255).astype('float32')
        Y_image_augmented = (scale_image(255*Y_image, scale)/255).astype('float32')
        Y_image_augmented[Y_image_augmented > 0.5] = 1.0
        Y_image_augmented[Y_image_augmented <= 0.5] = 0.0

        assert X_image_augmented.shape[0] == 256
        assert X_image_augmented.shape[1] == 256
        assert Y_image_augmented.shape[0] == 256
        assert Y_image_augmented.shape[1] == 256

        if debug:
            images_to_show = []
            titles = []
            images_to_show.append(np.flip(X_image, axis=-1))
            titles.append('Original image')
            images_to_show.append(Y_image)
            titles.append('Original GT image')
            images_to_show.append(np.flip((255*X_image_augmented).astype('uint8'), axis=-1))
            titles.append('Augmented X image')
            images_to_show.append(Y_image_augmented)
            titles.append('Augmented Y image')
            show_images(images_to_show, titles, (2, 2))

        ds_utils.np_to_tf_record(X_image_augmented, Y_image_augmented, filename.numpy(), output_dataset_writer)

print()
print('Total dataset elements: {}'.format(count))
print('Scale augmented elements: {}'.format(augmentation_count))
print('Requested/actual augmentation frequency: {}/{}'.format(augmentation_frequency, augmentation_count / count))


