import os
import argparse
import cv2

parser = argparse.ArgumentParser(description='Setup PHIBD 2012 dataset')
parser.add_argument('installation_path', help='Where to install the dataset')
args = parser.parse_args()

installation_path = args.installation_path

if not os.path.exists(installation_path):
    os.makedirs(installation_path)

if not os.path.exists('PHIBD2012/GT.zip'):
    os.system(
        'wget http://www.iapr-tc11.org/dataset/PHIBD2012/GT.zip')

os.system('unzip -o GT.zip')

if not os.path.exists('Original.zip'):
    os.system(
        'wget http://www.iapr-tc11.org/dataset/PHIBD2012/Original.zip')

os.system('unzip -o Original.zip')

subpath = 'phibd2012'
full_installation_original_path = os.path.join(
    installation_path, subpath, 'original')
full_installation_gt_path = os.path.join(installation_path, subpath, 'gt')
if not os.path.exists(full_installation_original_path):
    os.makedirs(full_installation_original_path)
if not os.path.exists(full_installation_gt_path):
    os.makedirs(full_installation_gt_path)

input_path = 'GT'
for filename in os.listdir(input_path):
    full_path = os.path.join(input_path, filename)
    basename, extension = os.path.splitext(filename)
    stem = basename[:9]
    full_output_path = os.path.join(
        full_installation_gt_path, f'{stem}.png')
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

input_path = 'Original'
for filename in os.listdir(input_path):
    full_path = os.path.join(input_path, filename)
    full_output_path = os.path.join(
        full_installation_original_path, filename)
    os.system(f'cp {full_path} {full_output_path}')

# Clean up
os.system('rm -rf Original')
os.system('rm -rf GT')
os.system('rm GT.zip')
os.system('rm Original.zip')
