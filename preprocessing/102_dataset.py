import cv2
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Parse 102 dataset')
parser.add_argument('--input-dir', help='Input directory', required=True)
parser.add_argument('--pattern', help='Channel filename pattern (example: F{}n.png', required=True)
args = parser.parse_args()
input_dir = args.input_dir.rstrip("/")
pattern = args.pattern

if not os.path.exists(input_dir) or not os.path.isdir(input_dir):
    print('Bad input directory')
    sys.exit(0)

b_filename = os.path.join(input_dir, pattern.format(2))
g_filename = os.path.join(input_dir, pattern.format(3))
r_filename = os.path.join(input_dir, pattern.format(4))

if not os.path.exists(r_filename) or not os.path.exists(g_filename) or not os.path.exists(b_filename):
    print('Directory contents not compatible')
    sys.exit(0)

r = cv2.imread(r_filename, cv2.IMREAD_GRAYSCALE)
g = cv2.imread(g_filename, cv2.IMREAD_GRAYSCALE)
b = cv2.imread(b_filename, cv2.IMREAD_GRAYSCALE)
merged = cv2.merge([b, g, r])
output_filename = '{}.png'.format(input_dir.split('/')[-1])
cv2.imwrite(output_filename, merged)