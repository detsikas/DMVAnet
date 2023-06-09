import os
import argparse
import cv2

parser = argparse.ArgumentParser(description='Setup H-DIBCO 2014 dataset')
parser.add_argument('installation_path', help='Where to install the dataset')
args = parser.parse_args()

installation_path = args.installation_path

if not os.path.exists(installation_path):
    os.makedirs(installation_path)

if not os.path.exists('GT.rar'):
    os.system(
        'wget https://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/dataset/GT.rar')

if not os.path.exists('HDIBCO2014_GT'):
    os.makedirs('HDIBCO2014_GT')
os.system('unrar x -o+ GT.rar HDIBCO2014_GT')

if not os.path.exists('original_images.rar'):
    os.system(
        'wget https://users.iit.demokritos.gr/~bgat/HDIBCO2014/benchmark/dataset/original_images.rar')
if not os.path.exists('HDIBCO2014_original_images'):
    os.makedirs('HDIBCO2014_original_images')
os.system('unrar x -o+ original_images.rar HDIBCO2014_original_images')

subpath = 'hdibco2014'
full_installation_original_path = os.path.join(
    installation_path, subpath, 'original')
full_installation_gt_path = os.path.join(installation_path, subpath, 'gt')
if not os.path.exists(full_installation_original_path):
    os.makedirs(full_installation_original_path)
if not os.path.exists(full_installation_gt_path):
    os.makedirs(full_installation_gt_path)

input_path = 'HDIBCO2014_GT'
for filename in os.listdir(input_path):
    if 'estGT' in filename:
        full_path = os.path.join(input_path, filename)
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        basename, extension = os.path.splitext(filename)
        stem = basename.split('_')[0]
        full_output_path = os.path.join(
            full_installation_gt_path, f'{stem}.jpg')
        cv2.imwrite(full_output_path, image)

input_path = 'HDIBCO2014_original_images'
for filename in os.listdir(input_path):
    basename, extension = os.path.splitext(filename)
    full_path = os.path.join(input_path, filename)
    full_output_path = os.path.join(
        full_installation_original_path, filename)
    os.system(f'cp {full_path} {full_output_path}')

# Clean up
os.system('rm -rf HDIBCO2014_*')
os.system('rm GT.rar')
os.system('rm original_images.rar')
