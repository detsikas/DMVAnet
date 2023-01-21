import numpy as np
import cv2
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Binarize a single image')
parser.add_argument('input_image', help='Input image')
args = parser.parse_args()

input_image = args.input_image

if not os.path.exists(input_image) or not os.path.isfile(input_image):
    print('Bad input image')
    sys.exit(0)

image = cv2.imread(input_image)
image[image < 127] = 0
image[image >= 127] = 255

basename = os.path.basename(input_image)
base, ext = os.path.splitext(basename)
output_filename = '{}_bin{}'.format(base, ext)
cv2.imwrite(output_filename, image)
