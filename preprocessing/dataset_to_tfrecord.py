import numpy as np
import argparse
import os
import sys
import tensorflow as tf
import preprocessing_dataset_utils as ds_utils

parser = argparse.ArgumentParser(description='Convert numpy files dataset to tfrecorddataset')
parser.add_argument('--dataset-training-path', required=True, help='Dataset training path')
parser.add_argument('--dataset-gt-path', required=True, help='Dataset ground truth path')
parser.add_argument('--output-dataset-file', required=True, help='Create dateset file')
args = parser.parse_args()

dataset_training_path = args.dataset_training_path
dataset_gt_path = args.dataset_gt_path
output_dataset_file = args.output_dataset_file

if not os.path.exists(dataset_training_path) or not os.path.isdir(dataset_training_path):
    print('Bad dataset training path')
    sys.exit(0)

if not os.path.exists(dataset_gt_path) or not os.path.isdir(dataset_gt_path):
    print('Bad dataset testing path')
    sys.exit(0)


output_dataset_writer = tf.io.TFRecordWriter(output_dataset_file)
training_files = [f for f in os.listdir(dataset_training_path) if os.path.isfile(os.path.join(dataset_training_path, f))]
gt_files = [f for f in os.listdir(dataset_gt_path) if os.path.isfile(os.path.join(dataset_gt_path, f))]
if len(training_files) != len(gt_files):
    print('Bad dataset. Mismatch between training and testing files (a)')
    sys.exit(0)

count = 0
for filename in training_files:
    print('Processing {}', filename, end='\r')
    X_file = os.path.join(dataset_training_path, filename)
    Y_file = os.path.join(dataset_gt_path, filename)
    if not os.path.exists(Y_file):
        print('\nBad dataset. Mismatch between training and testing files (b)')
        sys.exit(0)
    X = np.load(X_file)
    Y = np.load(Y_file)
    ds_utils.np_to_tf_record(X, Y, str.encode(filename), output_dataset_writer)
    count += 1

if count != len(training_files):
    print('\nSomething went wrong')

print('\n Done: {} files'.format(len(training_files)))

