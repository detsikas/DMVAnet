import argparse
import os
import sys
from dataset_utils import get_directory_depth
import re

# Input arguments
parser = argparse.ArgumentParser(description='Verify dataset folder structure')
parser.add_argument('dataset_source_path',
                    help='Path to root of dataset')
args = parser.parse_args()

dataset_source_path = args.dataset_source_path
depth = get_directory_depth(dataset_source_path)


def verify_subpath(subpath):
    print(f'Verifying {subpath}')
    folders = os.listdir(subpath)
    if 'original' not in folders or 'gt' not in folders or len(folders) != 2:
        print(f'Bad directory structure: {subpath}, {folders}')
        sys.exit(0)
    original_subpath = os.path.join(subpath, 'original')
    gt_subpath = os.path.join(subpath, 'gt')
    x_files = sorted(os.listdir(original_subpath))
    y_files = sorted(os.listdir(gt_subpath))
    number_of_x_files = len(x_files)
    number_of_y_files = len(y_files)
    if number_of_x_files != number_of_y_files:
        print(f'x, y number of files mismatch: {subpath}')
    for x, y in zip(x_files, y_files):
        x_nums = re.findall(r'\d+', x)
        y_nums = re.findall(r'\d+', y)
        if len(x_nums) != 1 or len(y_nums) != 1 or x_nums[0] != y_nums[0]:
            print('x, y numbering mismatch')
            sys.exit(0)


if depth == 1:
    verify_subpath(dataset_source_path)
else:
    folders = os.listdir(dataset_source_path)
    for folder in folders:
        verify_subpath(os.path.join(dataset_source_path, folder))

print('No error')
