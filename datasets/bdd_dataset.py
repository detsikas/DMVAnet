import cv2
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Parse bdd dataset gt images')
parser.add_argument('--input-dir', help='Input directory', required=True)
parser.add_argument('--output-dir', help='Output directory', required=True)
args = parser.parse_args()
input_dir = args.input_dir.rstrip("/")
output_dir = args.output_dir.rstrip("/")

if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
    print('Bad input directory')
    sys.exit(0)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Read images for prediction
filenames = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

for f in filenames:
    print('Processing {}\r'.format(f))
    full_filepath = os.path.join(input_dir, f)
    image = cv2.imread(full_filepath)
    assert len(image.shape) == 3

    image[image < 127] = 0
    image[image >= 127] = 255

    output_filename = os.path.join(output_dir, f)
    cv2.imwrite(output_filename, image)
