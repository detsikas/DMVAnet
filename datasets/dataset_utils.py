import tensorflow as tf
import os


RANDOM_GENERATOR = tf.random.Generator.from_non_deterministic_state()
NormalizationLayer = tf.keras.layers.Normalization(axis=-1, mean=[123.675, 116.28, 103.53],
                                                   variance=tf.math.square([58.395, 57.12, 57.375]))

rescale_m11 = tf.keras.layers.Rescaling(scale=1. / 127.5, offset=-1)
rescale_01 = tf.keras.layers.Rescaling(scale=1. / 255.0)


def preprocess_label(label):
    y = rescale_01(label)
    #y = tf.cast(y, tf.int32)
    return tf.cast(y, tf.float32)


def decode_dataset(path_ds, size=None):
    ds = path_ds.map(tf.io.read_file)
    ds = ds.map(tf.image.decode_image)

    return ds


def ensure_shape_for_single_image(images_, labels_):
    return tf.ensure_shape(images_, (None, None, 3)), tf.ensure_shape(labels_, (None, None, 3))


def preprocess_training_data(image, label, target_image_size, augment):
    seed = RANDOM_GENERATOR.make_seeds(2)[0]

    if target_image_size is not None:
        scale = tf.random.uniform([1], minval=0.75, maxval=2.0)[0]
        new_shape = [tf.cast(tf.cast(tf.shape(image)[0], tf.float32)*scale, tf.int32),
                     tf.cast(tf.cast(tf.shape(image)[1], tf.float32)*scale, tf.int32)]

        # For random transforms reference in torchvision
        # https://mmsegmentation.readthedocs.io/en/latest/_modules/mmseg/datasets/pipelines/transforms.html
        x = tf.image.resize(image, new_shape)
        y = tf.image.resize(label, new_shape)

        x = tf.image.stateless_random_crop(value=x, size=(target_image_size, target_image_size, x.shape[-1]),
                                           seed=seed)
        y = tf.image.stateless_random_crop(value=y, size=(target_image_size, target_image_size, y.shape[-1]),
                                           seed=seed)
    else:
        x = image
        y = label

    if augment:
        x = tf.image.stateless_random_flip_left_right(image=x, seed=seed)
        y = tf.image.stateless_random_flip_left_right(image=y, seed=seed)

        # Photometric distortion
        x = tf.cast(x, tf.uint8)
        x = tf.image.stateless_random_brightness(
            image=x, max_delta=32.0/255.0, seed=seed)  # torchvision sets to 32
        x = tf.clip_by_value(x, 0, 255)
        x = tf.cast(x, tf.float32)
        x = tf.image.stateless_random_contrast(
            image=x, lower=0.5, upper=1.5, seed=seed)
        x = tf.clip_by_value(x, 0, 255)
        x = tf.image.stateless_random_saturation(
            image=x, lower=0.5, upper=1.5, seed=seed)
        x = tf.clip_by_value(x, 0, 255)
        x = tf.image.stateless_random_hue(
            image=x, max_delta=18.0/180.0, seed=seed)  # torchvision sets 18
        x = tf.clip_by_value(x, 0, 255)

    # x = tf.image.per_image_standardization(x)
    # x = NormalizationLayer(x)

    return (rescale_m11(x), preprocess_label(y))


def remove_out_of_bounds_anchors(anchors, dimension, target_size):
    offset = 0
    for h in anchors:
        if h <= (dimension-target_size):
            offset += 1
        else:
            break

    return anchors[:offset]


def preprocess_inference_data(image, label):
    x = tf.cast(image, tf.float32)

    return (rescale_m11(x), preprocess_label(label))


def extract_inference_patches(image, target_image_size, stride):
    H = image.shape[0]
    W = image.shape[1]

    h_anchors = np.arange(0, H, stride)
    w_anchors = np.arange(0, W, stride)

    h_anchors = remove_out_of_bounds_anchors(h_anchors, H, target_image_size)
    w_anchors = remove_out_of_bounds_anchors(w_anchors, W, target_image_size)

    if image.shape[0]-target_image_size not in h_anchors:
        h_anchors = np.concatenate([h_anchors, [H-target_image_size]], axis=0)

    if image.shape[1]-target_image_size not in w_anchors:
        w_anchors = np.concatenate([w_anchors, [W-target_image_size]], axis=0)

    patches = []
    for h in h_anchors:
        for w in w_anchors:
            patch = image[h:h+target_image_size, w:w+target_image_size, :]
            patches.append(patch)

    x = tf.stack(patches)

    return x, h_anchors, w_anchors


def configure_dataset(dataset_, batch_size=1):
    if batch_size > 0:
        dataset_ = dataset_.batch(
            batch_size, num_parallel_calls=tf.data.AUTOTUNE)
    dataset_ = dataset_.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset_


def initialize_dataset(source_directory):
    x_path = os.path.join(
        source_directory, '*/original/*.*')
    y_path = os.path.join(
        source_directory, '*/gt/*.*')

    # Retrieve the x and y image filenames
    img_path_ds = tf.data.Dataset.list_files(
        file_pattern=x_path, shuffle=False)
    label_path_ds = tf.data.Dataset.list_files(
        file_pattern=y_path, shuffle=False)

    # Read and decode image files
    x_dataset = decode_dataset(img_path_ds)
    y_dataset = decode_dataset(label_path_ds)

    # Assemble the x and y images to one dataset
    dataset = tf.data.Dataset.zip((x_dataset, y_dataset))

    # Set dataset shape
    dataset = dataset.map(ensure_shape_for_single_image)

    return dataset


def create_dataset_training_pipeline(source_directory, batch_size, target_size, augment):
    # Initial dataset setup
    dataset = initialize_dataset(source_directory)

    # Preprocess data
    dataset = dataset.map(lambda image, label: preprocess_training_data(image, label, target_size, augment),
                          num_parallel_calls=tf.data.AUTOTUNE)

    dataset = configure_dataset(dataset, batch_size)

    return dataset


def create_dataset_inference_pipeline(source_directory):
    # Initial dataset setup
    dataset = initialize_dataset(source_directory)

    # Preprocess data
    dataset = dataset.map(lambda image, label: preprocess_inference_data(image, label),
                          num_parallel_calls=tf.data.AUTOTUNE)

    dataset = configure_dataset(dataset, 0)

    return dataset
