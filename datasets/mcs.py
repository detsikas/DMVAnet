import os
import argparse
import cv2

parser = argparse.ArgumentParser(
    description='Setup MCS (Monk Cuper Set) dataset')
parser.add_argument('installation_path', help='Where to install the dataset')
args = parser.parse_args()

installation_path = args.installation_path

if not os.path.exists(installation_path):
    os.makedirs(installation_path)

if not os.path.exists('MCSset.tar.gz'):
    os.system(
        'wget https://www.ai.rug.nl/~sheng/MCSset.tar.gz')

if not os.path.exists('mcs'):
    os.makedirs('mcs')

os.system('tar xvf MCSset.tar.gz --directory mcs')

subpath = 'mcs'
full_installation_original_path = os.path.join(
    installation_path, subpath, 'original')
full_installation_gt_path = os.path.join(installation_path, subpath, 'gt')
if not os.path.exists(full_installation_original_path):
    os.makedirs(full_installation_original_path)
if not os.path.exists(full_installation_gt_path):
    os.makedirs(full_installation_gt_path)

input_path = 'mcs'
for filename in os.listdir(input_path):
    if '.png' in filename:
        full_path = os.path.join(input_path, filename)
        basename, extension = os.path.splitext(filename)
        if 'GT-' in filename:
            filename = filename[3:]
            full_output_path = os.path.join(
                full_installation_gt_path, f'{basename}.png')
            image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
            _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
            cv2.imwrite(full_output_path, image)
        else:
            image = cv2.imread(full_path)
            full_output_path = os.path.join(
                full_installation_original_path, filename)
            cv2.imwrite(full_output_path, image)
# Clean up
os.system('rm -rf mcs')
os.system('rm MCSset.tar.gz')
