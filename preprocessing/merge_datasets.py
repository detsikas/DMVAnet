import argparse
import tensorflow as tf
import numpy as np
import os
import sys
from functools import partial
import preprocessing_dataset_utils as ds_utils

parser = argparse.ArgumentParser(description='Merge datasets')
parser.add_argument('--tfrecords', required=True, help='A list of tfrecord files', nargs='+')
parser.add_argument('--output-dataset-filename', required=True, help='Filename of the output datasaet')
args = parser.parse_args()

tfrecords = args.tfrecords
output_dataset_filename = args.output_dataset_filename

for tfr in tfrecords:
    if not os.path.exists(tfr):
        print('Bad tf record path')
        sys.exit(0)

dataset = tf.data.TFRecordDataset(tfrecords)
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

output_dataset_writer = tf.io.TFRecordWriter(output_dataset_filename)

count = 0
iterator = iter(parsed_dataset)
files_set = set()
for X_sample, Y_sample, filename in iterator:
    print('Processing {}', filename, end='\r')
    file_base = filename.numpy().decode().split('_')[0]
    files_set.add(file_base)
    ds_utils.np_to_tf_record(X_sample, Y_sample, filename.numpy(), output_dataset_writer)
    count += 1

print()
print('Distinct files in datasets: {}'.format(files_set))
