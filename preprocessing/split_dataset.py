import tensorflow as tf
import argparse
import os
import sys
from functools import partial
import preprocessing_dataset_utils as ds_utils

parser = argparse.ArgumentParser(description='Split a dataset into training and testing')
parser.add_argument('--tfrecords-dir', required=True, help='Tf record file to split')
parser.add_argument('--output-dir', required=True, help='Output path')
parser.add_argument('--training-ratio', default=0.8, help='Ratio of the training part (float between 0 and 1), default 0.8',
                    type=float)
args = parser.parse_args()
output_dir = args.output_dir
tfrecords_dir = args.tfrecords_dir
training_ratio = args.training_ratio

if not os.path.exists(tfrecords_dir) or not os.path.isdir(tfrecords_dir):
    print('Bad tf record directory')
    sys.exit(0)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)
elif not os.path.isdir(output_dir):
    print('Bad output dir')
    sys.exit(0)

dataset_files = [os.path.join(tfrecords_dir, f) for f in os.listdir(tfrecords_dir)
                 if os.path.isfile(os.path.join(tfrecords_dir, f)) and os.path.splitext(f)[1]=='.tfrecord']

for tfrecord in dataset_files:
    dataset = tf.data.TFRecordDataset(filenames=[tfrecord])

    parsed_dataset = dataset.map(partial(ds_utils.read_tf_record, with_filenames=True))

    dataset_size = ds_utils.count_records(parsed_dataset)
    train_dataset_size = int(training_ratio*dataset_size)
    print('Spliting {}'.format(tfrecord))
    print('Dataset size: {}'.format(dataset_size))
    print('Split ratio: {}'.format(training_ratio))
    print('Training size: {}'.format(train_dataset_size))
    print('Testing size: {}'.format(dataset_size-train_dataset_size))

    base, ext = os.path.splitext(os.path.basename(tfrecord))
    training_filename = os.path.join(output_dir, '{}_training{}'.format(base, ext))
    testing_filename = os.path.join(output_dir, '{}_testing{}'.format(base, ext))

    output_training_dataset_writer = tf.io.TFRecordWriter(training_filename)
    output_testing_dataset_writer = tf.io.TFRecordWriter(testing_filename)
    count = 0
    for X_sample, Y_sample, filename in parsed_dataset:
        if count < train_dataset_size:
            ds_utils.np_to_tf_record(X_sample.numpy(), Y_sample.numpy(), str.encode(filename.numpy().decode()), output_training_dataset_writer)
        else:
            ds_utils.np_to_tf_record(X_sample.numpy(), Y_sample.numpy(), str.encode(filename.numpy().decode()), output_testing_dataset_writer)
        count += 1



