import os
import argparse
import cv2

# original_gt_labeled.zip is required to be at the same folder as the script

parser = argparse.ArgumentParser(
    description='Setup Bickley diary dataset')
parser.add_argument('installation_path', help='Where to install the dataset')
args = parser.parse_args()

installation_path = args.installation_path

if not os.path.exists(installation_path):
    os.makedirs(installation_path)

os.system('unzip -o original_gt_labeled.zip')

subpath = 'bdd'
full_installation_original_path = os.path.join(
    installation_path, subpath, 'original')
full_installation_gt_path = os.path.join(installation_path, subpath, 'gt')
if not os.path.exists(full_installation_original_path):
    os.makedirs(full_installation_original_path)
if not os.path.exists(full_installation_gt_path):
    os.makedirs(full_installation_gt_path)

input_path = 'gt_labeled/original_gt'
for filename in os.listdir(input_path):
    full_path = os.path.join(input_path, filename)
    basename, extension = os.path.splitext(filename)
    stem = basename.split('_')[0]
    full_output_path = os.path.join(
        full_installation_gt_path, f'{stem}.bmp')
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(full_output_path, image)

input_path = 'gt_labeled/original'
for filename in os.listdir(input_path):
    full_path = os.path.join(input_path, filename)
    full_output_path = os.path.join(
        full_installation_original_path, filename)
    image = cv2.imread(full_path, cv2.IMREAD_COLOR)
    cv2.imwrite(full_output_path, image)

# Clean up
os.system('rm original_gt_labeled.zip')
os.system('rm -rf gt_labeled')
