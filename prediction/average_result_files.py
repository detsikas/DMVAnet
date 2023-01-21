import pandas as pd
import numpy as np
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Average excel files created from multiple image prediction repeated'
                                             ' multiple times')
parser.add_argument('--input-directory', help='Input directory', required=True)
parser.add_argument('--file-prefix', help='Output filename prefix', required=True)
args = parser.parse_args()

input_directory = args.input_directory
file_prefix = args.file_prefix

if not os.path.exists(input_directory) or not os.path.isdir(input_directory):
    print('Bad input directory')
    sys.exit(0)

# Read files for comparison
filenames = [f for f in os.listdir(input_directory)
             if os.path.isfile(os.path.join(input_directory, f)) and os.path.splitext(f)[-1] == '.xlsx']

print(filenames)
if len(filenames) == 0:
    print('No files to compare')
    sys.exit(0)

print(len(filenames))

filenames.sort()

results_dictionary = {'Metrics': ['MSE', 'BCE', 'BAC', 'F-measure', 'Precision', 'Recall', 'DRD', 'NRM', 'MPM', 'PSNR', 'Dice', 'mPSNR']}

dfs = []
for f in filenames:
    full_path = os.path.join(input_directory, f)
    dfs.append(pd.read_excel(full_path))

mean_df = pd.concat(dfs).groupby(level=0).mean()
var_df = pd.concat(dfs).groupby(level=0).var()
mean_df.to_excel('{}.xlsx'.format(file_prefix), index=False)
var_df.to_excel('var_{}.xlsx'.format(file_prefix), index=False)

