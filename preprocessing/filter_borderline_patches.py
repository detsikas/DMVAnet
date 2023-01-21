import tensorflow as tf
import argparse
import os
import sys
from functools import partial
import preprocessing_dataset_utils as ds_utils
from common.image_utils import show_images
import re


parser = argparse.ArgumentParser(description='Display the images of a tf record')
parser.add_argument('--tfrecord', required=True, help='tfrecord file')
parser.add_argument('--output-dataset-path', required=True, help='Path to put the output file')

args = parser.parse_args()
tfrecord_file = args.tfrecord
output_dataset_path = args.output_dataset_path

if not os.path.exists(tfrecord_file):
    print('Bad tf record path')
    sys.exit(0)

if not os.path.exists(output_dataset_path):
    os.mkdir(output_dataset_path)


dataset = tf.data.TFRecordDataset([tfrecord_file])
output_dataset_writer = tf.io.TFRecordWriter(os.path.join(output_dataset_path, os.path.basename(tfrecord_file)))
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


parsed_dataset = dataset.map(partial(ds_utils.read_tf_record, with_filenames=True))
removed_count = 0
include_count = 0
total_count = 0

# Find the edges
edges_x = {}
edges_y = {}
for X_sample, Y_sample, filename in parsed_dataset:
    #print('Displaying {}'.format(filename.numpy().decode()), end='\r')
    string_filename = filename.numpy().decode()
    basename, _ = os.path.splitext(string_filename)
    indices = [m.start() for m in re.finditer('_', basename)]
    prefix = basename[:indices[-2]]
    x = int(basename[indices[-2]+1:indices[-1]])
    y = int(basename[indices[-1]+1:])

    if prefix not in edges_x:
        edges_x[prefix] = x
    else:
        if x > edges_x[prefix]:
            edges_x[prefix] = x
            #print('{} x: {}'.format(prefix, x))

    if prefix not in edges_y:
        edges_y[prefix] = y
    else:
        if y > edges_y[prefix]:
            edges_y[prefix] = y
            #print('{} y: {}'.format(prefix, y))

    total_count += 1

for X_sample, Y_sample, filename in parsed_dataset:
    string_filename = filename.numpy().decode()
    basename, _ = os.path.splitext(string_filename)
    indices = [m.start() for m in re.finditer('_', basename)]
    prefix = basename[:indices[-2]]
    x = int(basename[indices[-2]+1:indices[-1]])
    y = int(basename[indices[-1]+1:])

    if edges_x[prefix] > x > 0 and edges_y[prefix] > y > 0:
        ds_utils.np_to_tf_record(X_sample, Y_sample, filename.numpy(), output_dataset_writer)
        include_count += 1
    else:
        removed_count += 1
        print('Removing {}'.format(string_filename))


print('\n Kept: {} files'.format(include_count))
print('\n Removed: {} files'.format(removed_count))
print('\n Total files: {}'.format(total_count))

print(edges_x)
print(edges_y)

print('\n Done')
