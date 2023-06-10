import os
import argparse
import cv2

# train.zip and train_cleaned.zip are required to be at the same folder as the script

parser = argparse.ArgumentParser(
    description='Setup Denoising Dirty Documents dataset')
parser.add_argument('installation_path', help='Where to install the dataset')
args = parser.parse_args()

installation_path = args.installation_path

if not os.path.exists(installation_path):
    os.makedirs(installation_path)

os.system('unzip -o train.zip')
os.system('unzip -o train_cleaned.zip')

subpath = 'ddd'
full_installation_original_path = os.path.join(
    installation_path, subpath, 'original')
full_installation_gt_path = os.path.join(installation_path, subpath, 'gt')
if not os.path.exists(full_installation_original_path):
    os.makedirs(full_installation_original_path)
if not os.path.exists(full_installation_gt_path):
    os.makedirs(full_installation_gt_path)

input_path = 'train_cleaned'
for filename in os.listdir(input_path):
    full_path = os.path.join(input_path, filename)
    full_output_path = os.path.join(
        full_installation_gt_path, filename)
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    cv2.imwrite(full_output_path, image)

input_path = 'train'
for filename in os.listdir(input_path):
    full_path = os.path.join(input_path, filename)
    full_output_path = os.path.join(
        full_installation_original_path, filename)
    image = cv2.imread(full_path, cv2.IMREAD_COLOR)
    cv2.imwrite(full_output_path, image)

# Clean up
os.system('rm -rf train')
os.system('rm -rf train_cleaned')
os.system('rm train.zip')
os.system('rm train_cleaned.zip')
