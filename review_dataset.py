import tensorflow as tf
import numpy as np
import argparse
import os
import common.dataset_utils as dataset_utils
from matplotlib import pyplot as plt
from PIL import Image
from matplotlib import image as mpimg

# Input arguments
parser = argparse.ArgumentParser(description='Review (train) dataset')
parser.add_argument('--target-image-size', type=int,
                    help='Desired image size before fine tuning (only square shapes allowed)', required=True)
parser.add_argument('--patch-size', type=int,
                    help='Patch size (default: 16)', default=16)
parser.add_argument('--classes', type=int,
                    help='Number of classes', required=True)
parser.add_argument('--dataset-source-path',
                    help='Path to root of cityscape dataset', required=True)
parser.add_argument(
    '--output-dir', help='If specified, images will be stored there')
parser.add_argument('--display-images',
                    help='Whether to display images',
                    action='store_true')
args = parser.parse_args()

target_image_size = args.target_image_size
patch_size = args.patch_size
number_of_classes = args.classes
dataset_source_path = args.dataset_source_path
output_dir = args.output_dir
display_images = args.display_images

if output_dir is not None and not os.path.exists(output_dir):
    os.makedirs(output_dir)

dataset = dataset_utils.create_dataset_training_pipeline(
    dataset_source_path, 'train', number_of_classes, 1, target_image_size, True)

i = 0
for item in dataset:
    image = item[0][0]
    gt = item[1][0]

    # Process image
    x = image+1.0
    x *= 127.5

    image_np = x.numpy()
    mean = np.mean(image_np, axis=(0, 1))
    stddev = np.std(image_np, axis=(0, 1))
    print(f'Mean: {mean}, stddev: {stddev}')

    # Process gt
    # x = tf.cast(tf.squeeze(label_), tf.int32)
    y_image = (gt.numpy()*255).astype('uint8')

    if output_dir is not None:
        filename = f'{i}_x.jpg'
        filename = os.path.join(output_dir, filename)
        Image.fromarray(image_np.astype('uint8')).save(filename)

        filename = f'{i}_y.jpg'
        filename = os.path.join(output_dir, filename)
        Image.fromarray(y_image.astype('uint8')).save(filename)

    if display_images:
        fig = plt.figure(figsize=(10, 7))

        fig.add_subplot(1, 2, 1)
        plt.imshow(image_np.astype('uint8'))
        plt.axis('off')
        plt.title("x")

        fig.add_subplot(1, 2, 2)
        plt.imshow(y_image.astype('uint8'))
        plt.axis('off')
        plt.title("y")
        plt.waitforbuttonpress()

    i += 1