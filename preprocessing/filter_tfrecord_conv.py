import argparse
import tensorflow as tf
import os
from common.image_utils import show_images
import numpy as np
from functools import partial
from scipy import signal
import preprocessing_dataset_utils as ds_utils

parser = argparse.ArgumentParser(description='Manually filter a tf record')
parser.add_argument('--tfrecord', required=True, help='Path to tf record file')
parser.add_argument('--output-dataset-path', required=True, help='Path to put the output file')
parser.add_argument('--display-images', help='Show the X, Y samples and the information percentage', action='store_true')
parser.add_argument('--window', help='Dimension of sliding window (default=100)', default=100)
parser.add_argument('--threshold', help='Threshold for filtering patches (default=0.9)',default=0.9, type=float)
args = parser.parse_args()

tfrecord_file = args.tfrecord
output_dataset_path = args.output_dataset_path
display_images = args.display_images
window = args.window
threshold = args.threshold

if not os.path.exists(tfrecord_file) or not os.path.isfile(tfrecord_file):
    print('Bad tf record file')

if not os.path.exists(output_dataset_path):
    os.mkdir(output_dataset_path)

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

output_dataset_writer = tf.io.TFRecordWriter(os.path.join(output_dataset_path, os.path.basename(tfrecord_file)))

count = 0
iterator = iter(parsed_dataset)
print('Threshold: {}'.format(threshold))
print('Window: {}'. format(window))

for X_sample, Y_sample, filename in iterator:
    print('Processing {}', filename)

    Y_np = Y_sample.numpy()
    mask = np.ones((window, window), dtype='float32') / (window*window)
    res = signal.convolve2d(Y_np, mask, mode='valid')
    value = np.min(res)
    print('Min ratio: {}'.format(value))
    print('Max ratio: {}'.format(np.max(res)))

    if value < threshold:
        print('Keeping patch')
        ds_utils.np_to_tf_record(X_sample, Y_sample, filename.numpy(), output_dataset_writer)

    if display_images:
        images_to_show = []
        titles = ['X']
        images_to_show.append((X_sample.numpy() * 255).astype('uint8'))

        grid = (1, 3)
        titles.append('Y')
        Y_gray = (Y_sample.numpy() * 255).astype('uint8')
        Y_rgb = np.stack((Y_gray, Y_gray, Y_gray))
        Y_rgb = np.moveaxis(Y_rgb, 0, -1)
        images_to_show.append(Y_rgb)
        titles.append('res')
        images_to_show.append((res*255).astype('uint8'))
        show_images(images_to_show, titles, grid, vmin=0, vmax=255, cmap='gray')

    count += 1

