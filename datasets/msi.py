import os
import argparse
import cv2

parser = argparse.ArgumentParser(description='Setup MSI dataset')
parser.add_argument('installation_path', help='Where to install the dataset')
args = parser.parse_args()

installation_path = args.installation_path

if not os.path.exists(installation_path):
    os.makedirs(installation_path)

os.system('unzip -o S_MSI_1_0.zip')
os.system('unzip -o S_MSI_2.zip')

pattern = 'F{}s.png'

# ---- 1 ----

subpath = 'msi_1'
full_installation_original_path = os.path.join(
    installation_path, subpath, 'original')
full_installation_gt_path = os.path.join(installation_path, subpath, 'gt')
if not os.path.exists(full_installation_original_path):
    os.makedirs(full_installation_original_path)
if not os.path.exists(full_installation_gt_path):
    os.makedirs(full_installation_gt_path)

input_path = 'S_MSI_1/GT/'
for filename in os.listdir(input_path):
    full_path = os.path.join(input_path, filename)
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    basename, extension = os.path.splitext(filename)
    stem = basename.split('G')[0]
    full_output_path = os.path.join(
        full_installation_gt_path, f'{stem}.png')
    cv2.imwrite(full_output_path, image)

input_path = 'S_MSI_1/MSI'
for filepath in os.listdir(input_path):
    output_filename = f'{filepath}.png'
    full_output_path = os.path.join(
        full_installation_original_path, filename)

    b_filename = os.path.join(input_path, filepath, pattern.format(2))
    g_filename = os.path.join(input_path, filepath, pattern.format(3))
    r_filename = os.path.join(input_path, filepath, pattern.format(4))

    r = cv2.imread(r_filename, cv2.IMREAD_GRAYSCALE)
    g = cv2.imread(g_filename, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(b_filename, cv2.IMREAD_GRAYSCALE)
    merged = cv2.merge([b, g, r])
    cv2.imwrite(os.path.join(
        full_installation_original_path, output_filename), merged)


# ---- 2 ----

subpath = 'msi_2'
full_installation_original_path = os.path.join(
    installation_path, subpath, 'original')
full_installation_gt_path = os.path.join(installation_path, subpath, 'gt')
if not os.path.exists(full_installation_original_path):
    os.makedirs(full_installation_original_path)
if not os.path.exists(full_installation_gt_path):
    os.makedirs(full_installation_gt_path)

input_path = 'S_MSI_2/GT/'
for filename in os.listdir(input_path):
    full_path = os.path.join(input_path, filename)
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    basename, extension = os.path.splitext(filename)
    stem = basename.split('G')[0]
    full_output_path = os.path.join(
        full_installation_gt_path, f'{stem}.png')
    cv2.imwrite(full_output_path, image)

input_path = 'S_MSI_2/MSI'
for filepath in os.listdir(input_path):
    output_filename = f'{filepath}.png'
    full_output_path = os.path.join(
        full_installation_original_path, filename)

    b_filename = os.path.join(input_path, filepath, pattern.format(2))
    g_filename = os.path.join(input_path, filepath, pattern.format(3))
    r_filename = os.path.join(input_path, filepath, pattern.format(4))

    r = cv2.imread(r_filename, cv2.IMREAD_GRAYSCALE)
    g = cv2.imread(g_filename, cv2.IMREAD_GRAYSCALE)
    b = cv2.imread(b_filename, cv2.IMREAD_GRAYSCALE)
    merged = cv2.merge([b, g, r])
    cv2.imwrite(os.path.join(
        full_installation_original_path, output_filename), merged)


# Clean up
os.system('rm -rf S_MSI_1*')
os.system('rm -rf S_MSI_2*')
