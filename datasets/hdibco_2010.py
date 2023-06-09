import os
import argparse
import cv2

parser = argparse.ArgumentParser(description='Setup H-DIBCO 2010 dataset')
parser.add_argument('installation_path', help='Where to install the dataset')
args = parser.parse_args()

installation_path = args.installation_path

if not os.path.exists(installation_path):
    os.makedirs(installation_path)

if not os.path.exists('H_DIBCO2010_GT.rar'):
    os.system(
        'wget https://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/H_DIBCO2010_GT.rar')

if not os.path.exists('H_DIBCO2010_GT'):
    os.makedirs('H_DIBCO2010_GT')
os.system('unrar x -o+ H_DIBCO2010_GT.rar H_DIBCO2010_GT')

if not os.path.exists('H_DIBCO2010_test_images.rar'):
    os.system(
        'wget https://users.iit.demokritos.gr/~bgat/H-DIBCO2010/benchmark/H_DIBCO2010_test_images.rar')
if not os.path.exists('H_DIBCO2010_test_images'):
    os.makedirs('H_DIBCO2010_test_images')
os.system('unrar x -o+ H_DIBCO2010_test_images.rar H_DIBCO2010_test_images')

subpath = 'hdibco2010'
full_installation_original_path = os.path.join(
    installation_path, subpath, 'original')
full_installation_gt_path = os.path.join(installation_path, subpath, 'gt')
if not os.path.exists(full_installation_original_path):
    os.makedirs(full_installation_original_path)
if not os.path.exists(full_installation_gt_path):
    os.makedirs(full_installation_gt_path)

input_path = 'H_DIBCO2010_GT'
for filename in os.listdir(input_path):
    if 'estGT' in filename:
        full_path = os.path.join(input_path, filename)
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        basename, extension = os.path.splitext(filename)
        stem = basename.split('_')[0]
        full_output_path = os.path.join(
            full_installation_gt_path, f'{stem}.png')
        cv2.imwrite(full_output_path, image)

input_path = 'H_DIBCO2010_test_images'
for filename in os.listdir(input_path):
    basename, extension = os.path.splitext(filename)
    full_path = os.path.join(input_path, filename)
    if extension == '.tif':
        image = cv2.imread(full_path)
        full_output_path = os.path.join(
            full_installation_original_path, f'{basename}.png')
        cv2.imwrite(full_output_path, image)
    else:
        full_output_path = os.path.join(
            full_installation_original_path, filename)
        os.system(f'cp {full_path} {full_output_path}')

# Clean up
os.system('rm -rf H_DIBCO2010_GT*')
os.system('rm -rf H_DIBCO2010_test_images*')
