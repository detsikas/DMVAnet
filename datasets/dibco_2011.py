import os
import argparse
import cv2

parser = argparse.ArgumentParser(description='Setup DIBCO 2011 dataset')
parser.add_argument('installation_path', help='Where to install the dataset')
args = parser.parse_args()

installation_path = args.installation_path

if not os.path.exists(installation_path):
    os.makedirs(installation_path)

if not os.path.exists('DIBCO11-handwritten.rar'):
    os.system(
        'wget http://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/dataset/DIBCO11-handwritten.rar')

if not os.path.exists('DIBCO11-handwritten'):
    os.makedirs('DIBCO11-handwritten')
os.system('unrar x -o+ DIBCO11-handwritten.rar DIBCO11-handwritten')

if not os.path.exists('DIBCO11-machine_printed.rar'):
    os.system(
        'wget http://utopia.duth.gr/~ipratika/DIBCO2011/benchmark/dataset/DIBCO11-machine_printed.rar')
if not os.path.exists('DIBCO11-machine_printed'):
    os.makedirs('DIBCO11-machine_printed')
os.system('unrar x -o+ DIBCO11-machine_printed.rar DIBCO11-machine_printed')

subpath = 'dibco2011'
full_installation_original_path = os.path.join(
    installation_path, subpath, 'original')
full_installation_gt_path = os.path.join(installation_path, subpath, 'gt')

if not os.path.exists(full_installation_original_path):
    os.makedirs(full_installation_original_path)
if not os.path.exists(full_installation_gt_path):
    os.makedirs(full_installation_gt_path)

input_path = 'DIBCO11-handwritten/'
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

input_path = 'DIBCO11-machine_printed/'
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
os.system('rm -rf DIBCO11-machine_printed*')
os.system('rm -rf DIBCO11-handwritten*')
