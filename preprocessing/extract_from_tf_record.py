import argparse
import tensorflow as tf
import os
import sys
from functools import partial

parser = argparse.ArgumentParser(description='Extract and describe a single tf record')
parser.add_argument('--tfrecord-file', required=True, help='Path to tf record file')
args = parser.parse_args()

tfrecord_file = args.tfrecord_file

if not os.path.exists(tfrecord_file) or not os.path.isfile(tfrecord_file):
    print('Bad tf record file')

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

    return X,Y


parsed_dataset = dataset.map(partial(_read_tf_record, with_filenames=True))
X_sample, Y_sample, filename = next(iter(parsed_dataset))
print(filename)
print('training dtype: {}'.format(X_sample.dtype))
print('training shape: {}'.format(X_sample.shape))
print('training type: {}'.format(type(X_sample)))

print('testing dtype: {}'.format(Y_sample.dtype))
print('testing shape: {}'.format(Y_sample.shape))
print('testing type: {}'.format(type(Y_sample)))
