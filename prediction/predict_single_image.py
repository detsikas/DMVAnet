import argparse
import os
import sys
import tensorflow as tf
import numpy as np
import cv2
import common.metrics as metrics
import common.wrappers as wrappers
import common.models as models
import common.model_info_io as model_info_io
import matplotlib.pyplot as plt
import common.image_utils as image_utils

# Image computations will be on binary 0,1 images


parser = argparse.ArgumentParser(description='Validate image')
parser.add_argument('--image-path', help='Path to image', required=True)
parser.add_argument('--model-path', help='Path to model', required=True)
parser.add_argument('--gt-image-path', help='Ground truth image')
parser.add_argument('--store-predicted-image', help='Store the predicted image at the same location', action='store_true')
parser.add_argument('--qualifier', help='Qualifier for the filename of the stored image', default='')
parser.add_argument('--show-images', help='Show original, predicted and ground truth images', action='store_true')
parser.add_argument('--border-type', help='Border type when padding images', choices=['replicate', 'white', 'black', 'reflect_101'],
                    default='reflect_101')
args = parser.parse_args()

image_path = args.image_path
model_path = args.model_path
gt_image_path = args.gt_image_path
store_predicted_image = args.store_predicted_image
qualifier = args.qualifier
with_show_images = args.show_images
border_type = args.border_type

if not os.path.exists(image_path) or not os.path.isfile(image_path):
    print('Bad input image')
    sys.exit(0)

if not os.path.exists(model_path) or not os.path.isdir(model_path):
    print('Bad model')
    sys.exit(0)

if gt_image_path is not None and (not os.path.exists(gt_image_path) or not os.path.isfile(gt_image_path)):
    print('Bad ground truth image path')
    sys.exit(0)


def pad_image(img, p_border_type):
    horizontal_remainder = img.shape[1] % 256
    vertical_remainder = img.shape[0] % 256
    horizontal = 0
    vertical = 0
    if horizontal_remainder != 0:
        horizontal = 256 - horizontal_remainder
    if vertical_remainder != 0:
        vertical = 256 - vertical_remainder
    left = horizontal // 2
    right = horizontal - left
    top = vertical // 2
    bottom = vertical - top
    if p_border_type == 'replicate':
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REPLICATE), top, bottom, left, right
    elif p_border_type == 'reflect_101':
        return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_REFLECT_101), top, bottom, left, right


def unpad_image(img, pad_top, pad_bottom, pad_left, pad_right):
    return img[pad_top:img.shape[0]-pad_bottom, pad_left:img.shape[1]-pad_right]


def extract_image_patches(img):
    i = 0
    j = 0
    extracted_patches = []
    while i < img.shape[0]-128:
        while j < img.shape[1]-128:
            patch = img[i:i + 256, j:j + 256]
            assert patch.shape[0] == 256
            assert patch.shape[1] == 256
            '''
            if debug:
                cv2.imshow('patch', patch)
                cv2.waitKey(0)
            '''
            patch = patch.astype(float)/255.0
            extracted_patches.append(patch)
            j += 128
        j = 0
        i += 128

    return extracted_patches


def reconstruct_image(image_patches, shape):
    i = 0
    j = 0
    k = 0
    reconstructed_image = np.zeros((shape[0], shape[1], 1), dtype=float)
    while i < reconstructed_image.shape[0]-128:
        while j < reconstructed_image.shape[1]-128:
            reconstructed_image[i:i + 256, j:j + 256] = image_patches[k]
            j += 128
            k += 1
        j = 0
        i += 128

    return reconstructed_image


def preprocess_image(img, p_border_type):
    padded_image, top, bottom, left, right = pad_image(img, p_border_type)
    return np.asarray(extract_image_patches(padded_image)), padded_image.shape, top, bottom, left, right


def show_image(img, title):
    img_to_show = np.copy(img)
    if np.max(img_to_show) <= 1.0 and img_to_show.dtype is not np.dtype('uint8'):
        img_to_show = metrics.threshold_image(img_to_show)*255
    fig = plt.figure()
    fig.suptitle(title)
    plt.imshow(img_to_show, cmap='gray')


image = cv2.imread(image_path)
#image = image_utils.remove_bg(cv2.imread(image_path), 0.8, True)
input_shape = [256, 256, 3]
model_info = model_info_io.read_info(os.path.join(model_path, 'info'))
model = model_info_io.restore_model(model_info)

print('Input shape: {}'.format(model_info.shape))
print('Model type: {}'.format(model_info.type))

model.load_weights(os.path.join(model_path, 'weights'))
patches, shape, pad_top, pad_bottom, pad_left, pad_right = preprocess_image(image, border_type)

# Predict
predictions = model.predict(patches)
predicted_image = np.squeeze(reconstruct_image(predictions, shape), axis=2)
predicted_image = unpad_image(predicted_image, pad_top, pad_bottom, pad_left, pad_right)
if gt_image_path is not None:
    gt_image = cv2.imread(gt_image_path)[:, :, 0] // 255

if with_show_images:
    show_image(predicted_image, 'Predicted_image')
    show_image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 'Original image')
    if gt_image_path is not None:
        show_image(gt_image, 'GT image')
    plt.show()

if store_predicted_image:
    output_path = os.path.dirname(image_path)
    filename = os.path.basename(image_path)
    base, ext = os.path.splitext(filename)
    output_filename = '{}_{}_predicted.png'.format(base, qualifier)
    cv2.imwrite(output_filename, metrics.threshold_image(predicted_image)*255, [cv2.IMWRITE_PNG_BILEVEL,1])

if gt_image_path is not None:
    mse = wrappers.mse(tf.cast(gt_image, tf.float32), predicted_image)
    bce = wrappers.bce(tf.cast(gt_image, tf.float32), tf.cast(predicted_image, tf.float32))
    bac = wrappers.bac(tf.cast(gt_image, tf.float32), tf.cast(predicted_image, tf.float32))
    f_measure, precision, recall = metrics.f1_score(tf.cast(tf.reshape(gt_image, [-1]), tf.uint8), metrics.threshold_image(tf.reshape(predicted_image, [-1])))
    p_f_measure, p_precision, p_recall = metrics.pfmeasure(gt_image, metrics.threshold_image(predicted_image))
    drd = metrics.drd_fn(gt_image.astype(int), metrics.threshold_image(predicted_image).astype(int))
    nrm = metrics.nrm(gt_image, predicted_image)
    mpm = metrics.mpm(gt_image, predicted_image)*1000
    psnr = wrappers.psnr(tf.cast(gt_image, tf.float32), tf.cast(metrics.threshold_image(predicted_image), tf.float32))
    manual_psnr = metrics.manual_psnr(gt_image.astype('float32'), predicted_image)
    dice = wrappers.dice(gt_image, predicted_image)
    print(
        'MSE: {}\nBCE: {}\nBAC: {}\nF-measure: {}\npF-measure: {}\nDRD: {}\nNRM: {}\nMPM: {}\nPSNR: {}\nDice: {}\nmPSNR: {}'.format(mse, bce, bac,
                                                                                                         f_measure, p_f_measure,
                                                                                                         drd,
                                                                                               nrm, mpm, psnr, dice, manual_psnr))
    print('P precision: {}'.format(p_precision))
    print('P recall: {}'.format(p_recall))
