import argparse
import tensorflow as tf
import os
from functools import partial
import preprocessing_dataset_utils as ds_utils
import numpy as np
import matplotlib.pyplot as plt
from common.noise_augmentation import gaussian_noise, sp_noise
from common.image_utils import show_images


parser = argparse.ArgumentParser(description='Augment the dataset by adding noise')
parser.add_argument('--tfrecord-file', required=True, help='Path to tf record file')
parser.add_argument('--gaussian-noise-augmentation-frequency', type=float, default=0.3,
                    help='How often to add gaussian noise')
parser.add_argument('--sp-noise-augmentation-frequency', type=float, default=0.3,
                    help='How often to add salt and pepper noise')
parser.add_argument('--gaussian-noise-variance', help="Gaussian noise variance", type=float, default=0.001)
parser.add_argument('--gaussian-noise-mean', help="Gaussian noise mean", type=float, default=0)
parser.add_argument('--sp-noise-prob', help='Salt & Pepper noise probability', type=float, default=0.1)
parser.add_argument('--output-path', help='Output path', required=True)
parser.add_argument('--debug', action='store_true', help='Show images')
args = parser.parse_args()

tfrecord_file = args.tfrecord_file
gaussian_noise_augmentation_frequency = args.gaussian_noise_augmentation_frequency
gaussian_noise_variance = args.gaussian_noise_variance
gaussian_noise_mean = args.gaussian_noise_mean
sp_noise_augmentation_frequency = args.sp_noise_augmentation_frequency
sp_noise_prob = args.sp_noise_prob
output_path = args.output_path
debug = args.debug

if not os.path.exists(tfrecord_file) or not os.path.isfile(tfrecord_file):
    print('Bad tf record file')

if not os.path.exists(output_path) or not os.path.isdir(output_path):
    os.makedirs(output_path)

basename = os.path.basename(tfrecord_file)
stem, ext = os.path.splitext(basename)
output_dataset_file = os.path.join(output_path, '{}_augment_noise{}'.format(stem, ext))

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
gaussian_noise_count = 0
sp_noise_count = 0

print('Gaussian Noise Augmentation by: {} {}'.format(gaussian_noise_mean, gaussian_noise_variance))
print('Salt & Pepper Noise Augmentation by: {}'.format(sp_noise_prob))

for X_sample, Y_sample, filename in iterator:
    count += 1

    gaussian_noise_augmentation_coin_toss_success = np.random.rand() <= gaussian_noise_augmentation_frequency
    sp_noise_augmentation_coin_toss_success = np.random.rand() <= sp_noise_augmentation_frequency

    if not gaussian_noise_augmentation_coin_toss_success and not sp_noise_augmentation_coin_toss_success:
        if np.random.rand() <= 0.5:
            gaussian_noise_augmentation_coin_toss_success = True
        else:
            sp_noise_augmentation_coin_toss_success = True

    X_image = X_sample.numpy()
    Y_image = Y_sample.numpy()

    if gaussian_noise_augmentation_coin_toss_success or sp_noise_augmentation_coin_toss_success:

        print('Augmenting {}', filename, end='\r')
        if debug:
            print('\n')
        augmentation_count += 1
        augmentation_level = 0
        X_image_augmented = np.copy(X_image)
        if gaussian_noise_augmentation_coin_toss_success:
            X_image_augmented = gaussian_noise(X_image_augmented, gaussian_noise_mean, gaussian_noise_variance)
            gaussian_noise_count += 1
        if sp_noise_augmentation_coin_toss_success:
            X_image_augmented = sp_noise(X_image_augmented, sp_noise_prob)
            sp_noise_count += 1
        if debug:
            images_to_show = []
            titles = []
            images_to_show.append(np.flip(X_image, axis=-1))
            titles.append('Original image')
            images_to_show.append(Y_image)
            titles.append('Original GT image')
            images_to_show.append(np.flip((255*X_image_augmented).astype('uint8'), axis=-1))
            titles.append('Augmented image')
            show_images(images_to_show, titles, (2, 2))
        ds_utils.np_to_tf_record(X_image_augmented, Y_image, filename.numpy(), output_dataset_writer)

print()
print('Total dataset elements: {}'.format(count))
print('Gaussian noise augmented elements: {}'.format(gaussian_noise_count))
print('Salt & Pepper noise augmented elements: {}'.format(sp_noise_count))
print('Augmentation frequency: {}'.format(augmentation_count/count))




