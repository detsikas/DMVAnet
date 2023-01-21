import tensorflow as tf


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    # If the value is an eager tensor BytesList won't unpack a string from an EagerTensor.
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def np_to_tf_record(X, Y, filename, writer):
    feature = {
        'X': _bytes_feature(tf.io.serialize_tensor(X)),
        'Y': _bytes_feature(tf.io.serialize_tensor(Y)),
        'filename': _bytes_feature(filename)
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())


feature_description = {
    'X': tf.io.FixedLenFeature([], tf.string),
    'Y': tf.io.FixedLenFeature([], tf.string),
    'filename': tf.io.FixedLenFeature([], tf.string)
}


def count_records(dataset):
    count = 0
    for _ in dataset:
        count += 1

    return count


def read_tf_record(example_proto, with_filenames=False):
    # Parse the input `tf.train.Example` proto using the dictionary above.
    raw_sample = tf.io.parse_single_example(example_proto, feature_description)
    X = tf.io.parse_tensor(raw_sample['X'], out_type=tf.float32)
    Y = tf.io.parse_tensor(raw_sample['Y'], out_type=tf.float32)
    if with_filenames:
        filename = raw_sample['filename']
        return X, Y, filename

    return X, Y


def dataset_to_file(dataset, output_dataset_file):
    output_dataset_writer = tf.io.TFRecordWriter(output_dataset_file)
    count = 0
    for X_sample, Y_sample, filename in dataset:
        np_to_tf_record(X_sample, Y_sample, str.encode(filename), output_dataset_writer)
        count += 1
    print('{} written cardinality: {}'.format(output_dataset_file, count))