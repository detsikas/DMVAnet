import os
import argparse
import cv2

parser = argparse.ArgumentParser(description='Setup DIBCO 2013 dataset')
parser.add_argument('installation_path', help='Where to install the dataset')
args = parser.parse_args()

installation_path = args.installation_path

if not os.path.exists(installation_path):
    os.makedirs(installation_path)

if not os.path.exists('DIBCO2013-dataset.rar'):
    os.system(
        'wget http://utopia.duth.gr/~ipratika/DIBCO2013/benchmark/dataset/DIBCO2013-dataset.rar')
if not os.path.exists('DIBCO2013-dataset'):
    os.makedirs('DIBCO2013-dataset')

os.system('unrar x -o+ DIBCO2013-dataset.rar DIBCO2013-dataset')

subpath = 'dibco2013'
full_installation_original_path = os.path.join(
    installation_path, subpath, 'original')
full_installation_gt_path = os.path.join(installation_path, subpath, 'gt')

if not os.path.exists(full_installation_original_path):
    os.makedirs(full_installation_original_path)
if not os.path.exists(full_installation_gt_path):
    os.makedirs(full_installation_gt_path)

input_path = 'DIBCO2013-dataset/GTimages'
for filename in os.listdir(input_path):
    full_path = os.path.join(input_path, filename)
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    basename, extension = os.path.splitext(filename)
    stem = basename.split('_')[0]
    full_output_path = os.path.join(
        full_installation_gt_path, f'{stem}.jpg')
    cv2.imwrite(full_output_path, image)

input_path = 'DIBCO2013-dataset/OriginalImages'
for filename in os.listdir(input_path):
    basename, extension = os.path.splitext(filename)
    full_path = os.path.join(input_path, filename)
    if extension == '.tiff':
        image = cv2.imread(full_path)
        full_output_path = os.path.join(
            full_installation_original_path, f'{basename}.jpg')
        cv2.imwrite(full_output_path, image)
    else:
        full_output_path = os.path.join(
            full_installation_original_path, filename)
        os.system(f'cp {full_path} {full_output_path}')

# Clean up
os.system('rm -rf DIBCO2013-dataset*')
