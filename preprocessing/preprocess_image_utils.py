import cv2
import os
import numpy as np


def scale_image(image):
    scaled_image_2 = cv2.resize(image, (0, 0), None, 0.5, 0.5, interpolation=cv2.INTER_AREA)
    scaled_image_4 = cv2.resize(image, (0, 0), None, 0.25, 0.25, interpolation=cv2.INTER_AREA)
    return scaled_image_2, scaled_image_4


def pad_image(image, border_type):
    horizontal_remainder = image.shape[1] % 256
    vertical_remainder = image.shape[0] % 256
    horizontal = 0
    vertical = 0
    if horizontal_remainder != 0:
        horizontal = 256 - horizontal_remainder
    if vertical_remainder != 0:
        vertical = 256 - vertical_remainder

    if border_type=='dump':
        if horizontal_remainder != 0 or vertical_remainder != 0:
            return None

    left = horizontal // 2
    right = horizontal - left
    top = vertical // 2
    bottom = vertical - top
    if border_type == 'replicate':
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REPLICATE)
    elif border_type == 'reflect_101':
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_REFLECT_101)
    else:
        return cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, None,
                                  (255, 255, 255) if border_type == 'white' else (0, 0, 0))


def extract_save_image_patches(image, output_file, debug=False):
    i = 0
    j = 0
    while i < image.shape[0] - 128:
        while j < image.shape[1] - 128:
            patch = image[i:i + 256, j:j + 256]
            assert patch.shape[0] == 256
            assert patch.shape[1] == 256
            if debug:
                cv2.imshow('patch', patch)
                cv2.waitKey(0)
            patch = patch.astype('float32') / 255.0
            base, ext = os.path.splitext(output_file)
            patch_filename = '{}_{}_{}.npy'.format(base, i, j)
            np.save(patch_filename, patch)
            j += 128
        j = 0
        i += 128


def extract_image_patches(image):
    i = 0
    j = 0
    patches = []
    while i < image.shape[0] - 128:
        while j < image.shape[1] - 128:
            patch = image[i:i + 256, j:j + 256]
            assert patch.shape[0] == 256
            assert patch.shape[1] == 256
            patches.append(patch)
            j += 128
        j = 0
        i += 128
    return patches
