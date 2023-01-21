import cv2
import argparse
import os
import sys
import preprocess_image_utils
import numpy as np

parser = argparse.ArgumentParser(description='Preprocess binarization dataset')
parser.add_argument('--input-path', help='Input directory', required=True)
parser.add_argument('--output-path', help='Output directory', required=True)
parser.add_argument('--border-type', help='Border type when padding images', choices=['replicate', 'white', 'black',
                                                                                      'reflect_101', 'dump'],
                    default='reflect_101')
parser.add_argument('--is_binary', help='Input images should be binary', action='store_true')
parser.add_argument('--with-scale', help='Apply scaling', action='store_true')
parser.add_argument('--debug', help='Debug functionality', action='store_true')
args = parser.parse_args()

input_path = args.input_path.rstrip("/")
output_path = args.output_path.rstrip("/")
border_type = args.border_type
is_binary= args.is_binary
with_scale = args.with_scale
debug = args.debug

if not os.path.exists(input_path) or not os.path.isdir(input_path):
    print('Bad input path')
    sys.exit(0)

if not os.path.exists(output_path):
    os.makedirs(output_path)
elif not os.path.isdir(output_path):
    print('Bad output path')
    sys.exit(0)


def splitall(path):
    allparts = []
    while 1:
        parts = os.path.split(path)
        if parts[0] == path:  # sentinel for absolute paths
            allparts.insert(0, parts[0])
            break
        elif parts[1] == path:  # sentinel for relative paths
            allparts.insert(0, parts[1])
            break
        else:
            path = parts[0]
            allparts.insert(0, parts[1])
    return allparts


def parse_directory(root_input_path, root_output_path):
    rejected_file_types = []
    for dirpath, dirnames, filenames in os.walk(root_input_path):
        output_path = dirpath.replace(root_input_path, root_output_path, 1)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        for filename in filenames:
            base, ext = os.path.splitext(filename)
            if ext == '.png' or ext == '.jpg' or ext == '.jpeg' or ext == '.tiff' or ext == '.bmp' or ext == '.tif':
                input_file = os.path.join(dirpath, filename)
                output_file = input_file.replace(dirpath, output_path)
                process_file(input_file, output_file)
            else:
                if ext not in rejected_file_types:
                    rejected_file_types.append(ext)

    print('Rejected file types:')
    for ext in rejected_file_types:
        print(ext)


def process_file(input_file, output_file):
    print('Processing {}', input_file)

    image = cv2.imread(input_file)

    # We only process color images
    assert len(image.shape) == 3

    if image is None:
        return

    if is_binary:
        assert(image.size-np.count_nonzero(image==0)-np.count_nonzero(image==255))==0
        assert(np.sum(image[:,:,0]-image[:,:,1]))==0
        assert (np.sum(image[:, :, 2] - image[:, :, 1])) == 0

    # Scale image
    scaled_image_2, scaled_image_4 = preprocess_image_utils.scale_image(image)
    print('\tShapes before padding: {} {} {}'.format(image.shape, scaled_image_2.shape, scaled_image_4.shape))
    # Pad images
    paded_image = preprocess_image_utils.pad_image(image, border_type)
    if paded_image is None:
        return
    base, ext = os.path.splitext(output_file)
    if with_scale:
        paded_image_2 = preprocess_image_utils.pad_image(scaled_image_2, border_type)
        paded_image_4 = preprocess_image_utils.pad_image(scaled_image_4, border_type)
        print('\tShapes after padding: {} {} {}'.format(paded_image.shape, paded_image_2.shape, paded_image_4.shape))
        if paded_image_2 is not None:
            stored_file = '{}_2{}'.format(base, ext)
            preprocess_image_utils.extract_save_image_patches(paded_image_2, stored_file)
        if paded_image_4 is not None:
            stored_file = '{}_4{}'.format(base, ext)
            preprocess_image_utils.extract_save_image_patches(paded_image_4, stored_file)

    # Extract and save image patches
    stored_file = '{}_1{}'.format(base, ext)
    preprocess_image_utils.extract_save_image_patches(paded_image, stored_file)


print('Border type: {}'.format(border_type))
parse_directory(input_path, output_path)
print('\nDone')
