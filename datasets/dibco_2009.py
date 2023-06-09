import os
import argparse
import cv2

parser = argparse.ArgumentParser(description='Setup DIBCO 2009 dataset')
parser.add_argument('installation_path', help='Where to install the dataset')
args = parser.parse_args()

installation_path = args.installation_path

if not os.path.exists(installation_path):
    os.makedirs(installation_path)

if not os.path.exists('DIBC02009_Test_images-handwritten.rar'):
    os.system('wget https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBC02009_Test_images-handwritten.rar')
os.system('unrar x -o+ DIBC02009_Test_images-handwritten.rar')

if not os.path.exists('DIBCO2009-GT-Test-images_handwritten.rar'):
    os.system('wget https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBCO2009-GT-Test-images_handwritten.rar')
os.system('unrar x -o+ DIBCO2009-GT-Test-images_handwritten.rar')

if not os.path.exists('DIBCO2009-GT-Test-images_printed.rar'):
    os.system('wget https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBCO2009-GT-Test-images_printed.rar')
os.system('unrar x -o+ DIBCO2009-GT-Test-images_printed.rar')

if not os.path.exists('DIBCO2009_Test_images-printed.rar'):
    os.system(
        'wget https://users.iit.demokritos.gr/~bgat/DIBCO2009/benchmark/DIBCO2009_Test_images-printed.rar')
os.system('unrar x -o+ DIBCO2009_Test_images-printed.rar')

subpath = 'dibco2009'
full_installation_original_path = os.path.join(
    installation_path, subpath, 'original')
full_installation_gt_path = os.path.join(installation_path, subpath, 'gt')
if not os.path.exists(full_installation_original_path):
    os.makedirs(full_installation_original_path)
if not os.path.exists(full_installation_gt_path):
    os.makedirs(full_installation_gt_path)

input_path = 'DIBC02009_Test_images-handwritten'
for filename in os.listdir(input_path):
    full_path = os.path.join(input_path, filename)
    full_output_path = os.path.join(
        full_installation_original_path, filename)
    os.system(f'cp {full_path} {full_output_path}')

input_path = 'DIBCO2009-GT-Test-images_handwritten'
for filename in os.listdir(input_path):
    full_path = os.path.join(input_path, filename)
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    basename, extension = os.path.splitext(filename)
    full_output_path = os.path.join(
        full_installation_gt_path, f'{basename}.png')
    cv2.imwrite(full_output_path, image)

input_path = 'DIBCO2009_Test_images-printed'
for filename in os.listdir(input_path):
    full_path = os.path.join(input_path, filename)
    full_output_path = os.path.join(
        full_installation_original_path, filename)
    os.system(f'cp {full_path} {full_output_path}')

input_path = 'DIBCO2009-GT-Test-images_printed'
for filename in os.listdir(input_path):
    full_path = os.path.join(input_path, filename)
    image = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
    _, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    basename, extension = os.path.splitext(filename)
    full_output_path = os.path.join(
        full_installation_gt_path, f'{basename}.png')
    cv2.imwrite(full_output_path, image)


# Clean up
os.system('rm -rf DIBC02009_Test_images-handwritten*')
os.system('rm -rf DIBCO2009-GT-Test-images_handwritten*')
os.system('rm -rf DIBCO2009_Test_images-printed*')
os.system('rm -rf DIBCO2009-GT-Test-images_printed*')
