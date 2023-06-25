import os
import argparse
import cv2

parser = argparse.ArgumentParser(description='Setup DIBCO 2019 dataset')
parser.add_argument('installation_path', help='Where to install the dataset')
args = parser.parse_args()

installation_path = args.installation_path

if not os.path.exists(installation_path):
    os.makedirs(installation_path)

if not os.path.exists('dibco2019_gt_trackA.zip'):
    os.system(
        'wget https://vc.ee.duth.gr/dibco2019/benchmark/dibco2019_gt_trackA.zip')

os.system('unzip -o dibco2019_gt_trackA.zip')

if not os.path.exists('dibco2019_GT_trackB.zip'):
    os.system(
        'wget https://vc.ee.duth.gr/dibco2019/benchmark/dibco2019_GT_trackB.zip')

os.system('unzip -o dibco2019_GT_trackB.zip')

if not os.path.exists('dibco2019_dataset_trackA.zip'):
    os.system(
        'wget https://vc.ee.duth.gr/dibco2019/benchmark/dibco2019_dataset_trackA.zip')

os.system('unzip -o dibco2019_dataset_trackA.zip')

if not os.path.exists('dibco2019_dataset_trackB.zip'):
    os.system(
        'wget https://vc.ee.duth.gr/dibco2019/benchmark/dibco2019_dataset_trackB.zip')

os.system('unzip -o dibco2019_dataset_trackB.zip')

subpath = 'dibco2019'
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
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    basename, extension = os.path.splitext(filename)
    stem = basename.split('_')[0]
    full_output_path = os.path.join(
        full_installation_gt_path, f'{stem}.png')
    cv2.imwrite(full_output_path, image)

input_path = 'Dataset'
for filename in os.listdir(input_path):
    basename, extension = os.path.splitext(filename)
    if extension == '.bmp':
        full_path = os.path.join(input_path, filename)
        full_output_path = os.path.join(
            full_installation_original_path, filename)
        os.system(f'cp {full_path} {full_output_path}')

# Clean up
os.system('rm -rf GT')
os.system('rm -rf Dataset')
os.system('rm dibco2019*.zip')
