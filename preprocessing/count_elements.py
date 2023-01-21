import numpy as np
import argparse
import os
import sys
import tensorflow as tf

parser = argparse.ArgumentParser(description='Count tf record entries')
parser.add_argument('--tfrecord', required=True, help='Tf record file')
args = parser.parse_args()

tfrecord = args.tfrecord

if not os.path.exists(tfrecord) or not os.path.isfile(tfrecord):
    print('Bad tfrecord file')
    sys.exit(0)


def count_records(dataset):
    count = 0
    for sample in dataset:
        count+=1

    return count


dataset = tf.data.TFRecordDataset(filenames=[tfrecord])
count = count_records(dataset)
print('{} elements in tfrecord {}'.format(count, tfrecord))
