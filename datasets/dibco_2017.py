import os
import argparse
import cv2

parser = argparse.ArgumentParser(description='Setup DIBCO 2017 dataset')
parser.add_argument('installation_path', help='Where to install the dataset')
args = parser.parse_args()

installation_path = args.installation_path

if not os.path.exists(installation_path):
    os.makedirs(installation_path)

if not os.path.exists('DIBCO2017_Dataset.7z'):
    os.system(
        'wget http://vc.ee.duth.gr/dibco2017/benchmark/DIBCO2017_Dataset.7z')

os.system('7z x DIBCO2017_Dataset.7z -y')

if not os.path.exists('DIBCO2017_GT.7z'):
    os.system(
        'wget http://vc.ee.duth.gr/dibco2017/benchmark/DIBCO2017_GT.7z')

os.system('7z x DIBCO2017_GT.7z -y')

subpath = 'dibco2017'
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
    image = cv2.imread(full_path)
    basename, extension = os.path.splitext(filename)
    stem = basename.split('_')[0]
    full_output_path = os.path.join(
        full_installation_gt_path, f'{stem}{extension}')
    os.system(f'cp {full_path} {full_output_path}')

input_path = 'Dataset'
for filename in os.listdir(input_path):
    basename, extension = os.path.splitext(filename)
    full_path = os.path.join(input_path, filename)
    full_output_path = os.path.join(
        full_installation_original_path, filename)
    os.system(f'cp {full_path} {full_output_path}')

# Clean up
os.system('rm -rf Dataset*')
os.system('rm -rf GT')
os.system('rm -rf DIBCO2017*')
