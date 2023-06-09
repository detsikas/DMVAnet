import os
import argparse
import cv2

parser = argparse.ArgumentParser(description='Setup DIBCO 2016 dataset')
parser.add_argument('installation_path', help='Where to install the dataset')
args = parser.parse_args()

installation_path = args.installation_path

if not os.path.exists(installation_path):
    os.makedirs(installation_path)

if not os.path.exists('DIBCO2016_dataset-GT.zip'):
    os.system(
        'wget http://vc.ee.duth.gr/h-dibco2016/benchmark/DIBCO2016_dataset-GT.zip')

os.system('unzip -o DIBCO2016_dataset-GT.zip')

if not os.path.exists('DIBCO2016_dataset-original.zip'):
    os.system(
        'wget http://vc.ee.duth.gr/h-dibco2016/benchmark/DIBCO2016_dataset-original.zip')

os.system('unzip -o DIBCO2016_dataset-original.zip')

subpath = 'dibco2016'
full_installation_original_path = os.path.join(
    installation_path, subpath, 'original')
full_installation_gt_path = os.path.join(installation_path, subpath, 'gt')
if not os.path.exists(full_installation_original_path):
    os.makedirs(full_installation_original_path)
if not os.path.exists(full_installation_gt_path):
    os.makedirs(full_installation_gt_path)

input_path = 'DIPCO2016_Dataset_GT'
for filename in os.listdir(input_path):
    full_path = os.path.join(input_path, filename)
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    basename, extension = os.path.splitext(filename)
    stem = basename.split('_')[0]
    full_output_path = os.path.join(
        full_installation_gt_path, f'{stem}.jpg')
    cv2.imwrite(full_output_path, image)

input_path = 'DIPCO2016_dataset'
for filename in os.listdir(input_path):
    basename, extension = os.path.splitext(filename)
    full_path = os.path.join(input_path, filename)
    full_output_path = os.path.join(
        full_installation_original_path, filename)
    os.system(f'cp {full_path} {full_output_path}')

# Clean up
os.system('rm -rf DIBCO2016*')
os.system('rm -rf DIPCO2016*')
