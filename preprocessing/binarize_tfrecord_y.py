import argparse
import tensorflow as tf
import os
from functools import partial
import preprocessing_dataset_utils as ds_utils

parser = argparse.ArgumentParser(description='Binarize the Y samples of the tfrecord')
tfrecord_input_group = parser.add_mutually_exclusive_group(required=True)
tfrecord_input_group.add_argument('--tfrecord', help='Path to tf record file')
tfrecord_input_group.add_argument('--tfrecord-dir', help='Directory to recursively read tfrecord from')
parser.add_argument('--output-dataset-path', required=True, help='Path to put the output file')
parser.add_argument('--threshold', help='Binarization threshold 0...1 (default=0.7)', default=0.7)
args = parser.parse_args()

tfrecord_file = args.tfrecord
tfrecord_dir = args.tfrecord_dir
output_dataset_path = args.output_dataset_path
threshold = args.threshold

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


def process_tf_record(tfrecord_file_param, output_path):
    dataset = tf.data.TFRecordDataset([tfrecord_file_param])
    parsed_dataset = dataset.map(partial(_read_tf_record, with_filenames=True))
    output_dataset_writer = tf.io.TFRecordWriter(
        os.path.join(output_path, os.path.basename(tfrecord_file_param)))

    count = 0
    iterator = iter(parsed_dataset)
    for X_sample, Y_sample, filename in iterator:
        print('Processing {}', filename, end='\r')
        Y = Y_sample.numpy()
        Y[Y >= threshold] = 1
        Y[Y < threshold] = 0
        ds_utils.np_to_tf_record(X_sample, Y, filename.numpy(), output_dataset_writer)
        count += 1

    print()


if tfrecord_file is not None:
    if not os.path.exists(tfrecord_file) or not os.path.isfile(tfrecord_file):
        print('Bad tf record file')

    if not os.path.exists(output_dataset_path):
        os.mkdir(output_dataset_path)
    process_tf_record(tfrecord_file, output_dataset_path)
elif tfrecord_dir is not None:
    if not os.path.exists(tfrecord_dir) or not os.path.isdir(tfrecord_dir):
        print('Bad tf record directory')

    for root, dirs, files in os.walk(tfrecord_dir):
        for f in files:
            basename, ext = os.path.splitext(f)
            if ext == '.tfrecord':
                print('Reading from: {}'.format(root))
                replaced_root = root.replace(tfrecord_dir, os.path.join(output_dataset_path, ''))
                print('Writing to: {}'.format(replaced_root))
                if not os.path.exists(replaced_root):
                    os.mkdir(replaced_root)
                process_tf_record(os.path.join(root, f), replaced_root)


