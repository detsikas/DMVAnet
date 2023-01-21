import argparse
import tensorflow as tf
import os
import sys
from functools import partial
import preprocessing_dataset_utils as ds_utils
import cv2
import numpy as np

parser = argparse.ArgumentParser(description='Filter tf records with variance')
parser.add_argument('--tfrecord-file', required=True, help='Path to tf record file')
parser.add_argument('--output-dataset-path', required=True, help='Path to put the output file')
parser.add_argument('--threshold', help='Variance threshold', type=float, required=True)
args = parser.parse_args()

tfrecord_file = args.tfrecord_file
output_dataset_path = args.output_dataset_path
threshold = args.threshold

if not os.path.exists(tfrecord_file) or not os.path.isfile(tfrecord_file):
    print('Bad tf record file')
    sys.exit(0)

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

    return X,Y


parsed_dataset = dataset.map(partial(_read_tf_record, with_filenames=True))

output_dataset_writer = tf.io.TFRecordWriter(os.path.join(output_dataset_path, os.path.basename(tfrecord_file)))

count = 0
total_count = 0
iterator = iter(parsed_dataset)
for X_sample, Y_sample, filename in iterator:
    print('Processing {}', filename, end='\r')
    hsv = cv2.cvtColor(X_sample.numpy()*255.0, cv2.COLOR_BGR2HSV)
    variance = np.var(hsv[:,:,2])
    if variance>threshold:
        ds_utils.np_to_tf_record(X_sample, Y_sample, filename.numpy(), output_dataset_writer)
        count += 1
    total_count += 1

print('\n Kept: {} files'.format(count))
print('\n Total files: {}'.format(total_count))

