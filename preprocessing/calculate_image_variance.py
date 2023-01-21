import cv2
import argparse
import os
import sys
import numpy as np
import preprocess_image_utils

parser = argparse.ArgumentParser(description='Calculate image variance')
parser.add_argument('--input-image', help='Input image', required=True)
args = parser.parse_args()

input_image = args.input_image

if not os.path.exists(input_image) or not os.path.isfile(input_image):
    print('Bad input image')
    sys.exit(0)

img = cv2.imread(input_image)
padded_img = preprocess_image_utils.pad_image(img, 'replicate')
patches = preprocess_image_utils.extract_image_patches(padded_img)
for patch in patches:
    variance = np.var(patch)
    float_variance = np.var(patch.astype(float)/255.0)
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    v_variance = np.var(hsv[:, :, 2])
    s_variance = np.var([hsv[:,:,1]])
    print('Variance: {}, float variance: {}, s variance: {}, v varnace: {}'.format(variance, float_variance,
                                                                                   s_variance, v_variance))
    cv2.imshow('test', patch)
    cv2.waitKey(0)
