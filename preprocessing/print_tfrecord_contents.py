import tensorflow as tf
import argparse
import os
import sys
from functools import partial

parser = argparse.ArgumentParser(description='Verify the contents of a tf record')
parser.add_argument('--tfrecords', required=True, help='A list of tfrecord files', nargs='+')
parser.add_argument('--filenames_only', action='store_true')
args = parser.parse_args()

tfrecords = args.tfrecords
filenames_only = args.filenames_only

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


count = 0
parsed_dataset = dataset.map(partial(_read_tf_record, with_filenames=True))
files_set = set()
for X_sample, Y_sample, filename in parsed_dataset:
    count += 1
    filename_str = filename.numpy().decode()
    file_base = filename_str.split('_')[0]
    files_set.add(file_base)
    if not filenames_only:
        print('{} {} {}'.format(filename_str, X_sample.shape, Y_sample.shape))

print('Found {} records'.format(count))

files_list = list(files_set)
files_list.sort()
print('Distinct files in dataset: {}'.format(files_list))
print('Count: {} distinct files'.format(len(list(files_list))))


