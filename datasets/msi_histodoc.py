import os
import argparse
import cv2
import re

# HISTODOC1.zip is required to be at the same folder as the script

parser = argparse.ArgumentParser(description='Setup MSI dataset')
parser.add_argument('installation_path', help='Where to install the dataset')
args = parser.parse_args()

installation_path = args.installation_path

if not os.path.exists(installation_path):
    os.makedirs(installation_path)

os.system('unzip -o HISTODOC1.zip')

pattern = 'F{}n.png'

subpath = 'msi_histodoc'
full_installation_original_path = os.path.join(
    installation_path, subpath, 'original')
full_installation_gt_path = os.path.join(installation_path, subpath, 'gt')
if not os.path.exists(full_installation_original_path):
    os.makedirs(full_installation_original_path)
if not os.path.exists(full_installation_gt_path):
    os.makedirs(full_installation_gt_path)

input_path = 'HISTODOC1'
for filepath in os.listdir(input_path):
    full_path = os.path.join(input_path, filepath)
    # GT
    number = re.findall(r'\d+', filepath)[0]
    file_prefix = f'z{number}'
    input_file = os.path.join(full_path, f'{file_prefix}GT.png')
    if not os.path.exists(input_file):
        input_file = os.path.join(full_path, f'{file_prefix}GTn.png')
    if not os.path.exists(input_file):
        input_file = os.path.join(full_path, filepath, f'{file_prefix}GT.png')
    image = cv2.imread(input_file, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    full_output_path = os.path.join(
        full_installation_gt_path, f'{file_prefix}.png')
    cv2.imwrite(full_output_path, image)

    # X
    if not os.path.exists(os.path.join(input_path, filepath, pattern.format(2))):
        filepath = os.path.join(filepath, filepath)
    b_filename = os.path.join(input_path, filepath, pattern.format(2))
    g_filename = os.path.join(input_path, filepath, pattern.format(3))
    r_filename = os.path.join(input_path, filepath, pattern.format(4))

    r = cv2.imread(r_filename, cv2.IMREAD_GRAYSCALE)
    g = cv2.imread(g_filename, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(b_filename, cv2.IMREAD_GRAYSCALE)
    merged = cv2.merge([b, g, r])
    cv2.imwrite(os.path.join(
        full_installation_original_path, f'{file_prefix}.png'), merged)

# Clean up
os.system('rm -rf HISTODOC1*')
