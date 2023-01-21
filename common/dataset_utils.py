import tensorflow as tf
import os
import sys
from functools import partial

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


def configure_dataset(dataset, shuffle_size, batch_size):
    dataset = dataset.shuffle(shuffle_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)

    return dataset


def count_records(dataset):
    count = 0
    for sample in dataset:
        count+=1

    return count


# split_percentage is a value between 0.0 and 1.0 indicating the portion of the trainign test size
def get_dataset(dataset_files, dataset_dir, shuffle_size, batch_size):
    if dataset_files is None:
        dataset_files = []
        if not os.path.exists(dataset_dir) or not os.path.isdir(dataset_dir):
            print('Bad dataset directory')
            sys.exit(0)
        for root, dirs, files in os.walk(dataset_dir, followlinks=True):
            for f in files:
                basename, ext = os.path.splitext(f)
                if ext == '.tfrecord':
                    dataset_files.append(os.path.join(root, f))

    for file in dataset_files:
        if not os.path.exists(file) or not os.path.isfile(file):
            print('Bad root data path')
            sys.exit(0)
        print(file)

    dataset = tf.data.TFRecordDataset(filenames=dataset_files)
    parsed_dataset = dataset.map(partial(_read_tf_record))
    return configure_dataset(parsed_dataset, shuffle_size, batch_size)
