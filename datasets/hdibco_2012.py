import os
import argparse
import cv2

parser = argparse.ArgumentParser(description='Setup H-DIBCO 2012 dataset')
parser.add_argument('installation_path', help='Where to install the dataset')
args = parser.parse_args()

installation_path = args.installation_path

if not os.path.exists(installation_path):
    os.makedirs(installation_path)

if not os.path.exists('H-DIBCO2012-dataset.rar'):
    os.system(
        'wget http://utopia.duth.gr/~ipratika/HDIBCO2012/benchmark/dataset/H-DIBCO2012-dataset.rar')

os.system('unrar x -o+ H-DIBCO2012-dataset.rar')

subpath = 'hdibco2012'
full_installation_original_path = os.path.join(
    installation_path, subpath, 'original')
full_installation_gt_path = os.path.join(installation_path, subpath, 'gt')

if not os.path.exists(full_installation_original_path):
    os.makedirs(full_installation_original_path)
if not os.path.exists(full_installation_gt_path):
    os.makedirs(full_installation_gt_path)

input_path = 'H-DIBCO2012-dataset/H-DIBCO2012-dataset'
for filename in os.listdir(input_path):
    full_path = os.path.join(input_path, filename)
    if '_GT' in filename:
        image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
        _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        basename, extension = os.path.splitext(filename)
        stem = basename.split('_')[0]
        full_output_path = os.path.join(
            full_installation_gt_path, f'{stem}.png')
        cv2.imwrite(full_output_path, image)
    else:
        full_output_path = os.path.join(
            full_installation_original_path, filename)
        os.system(f'cp {full_path} {full_output_path}')

# Clean up
os.system('rm -rf H-DIBCO2012-dataset*')
