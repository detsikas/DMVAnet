import pandas as pd
import numpy as np
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Compare excel files created from multiple image prediction')
parser.add_argument('--input-directory', help='Input directory', required=True)
parser.add_argument('--qualifier', help='Output qualifier')
args = parser.parse_args()

input_directory = args.input_directory
qualifier = args.qualifier

if not os.path.exists(input_directory) or not os.path.isdir(input_directory):
    print('Bad input directory')
    sys.exit(0)

# Read files for comparison
filenames = [f for f in os.listdir(input_directory)
             if os.path.isfile(os.path.join(input_directory, f)) and os.path.splitext(f)[-1] == '.xlsx']


if len(filenames) == 0:
    print('No files to compare')
    sys.exit(0)

filenames.sort()

results_dictionary = {'Metrics': ['MSE', 'BCE', 'BAC', 'F-measure', 'Precision', 'Recall', 'DRD', 'NRM', 'MPM', 'PSNR', 'Dice', 'mPSNR']}

for f in filenames:
    full_path = os.path.join(input_directory, f)
    in_df = pd.read_excel(full_path)
    mean_results = in_df['mean']
    results_dictionary[f] = mean_results


df = pd.DataFrame(data=results_dictionary)
df.set_index('Metrics', inplace=True)
array_fm = df.loc['F-measure'].to_numpy()
array_drd = df.loc['DRD'].to_numpy()
array_PSNR = df.loc['PSNR'].to_numpy()


def measurements_to_score(array, _reversed=False):
    indices = np.argsort(array)
    number_of_measurements = array.shape[0]
    score_array = np.zeros(number_of_measurements, dtype=int)

    for i in range(number_of_measurements):
        location = indices[i]
        score_array[location] = i

    if _reversed:
        score_array = number_of_measurements-1-score_array
    return score_array


score_fm = measurements_to_score(array_fm)
score_drd = measurements_to_score(array_drd, True)
score_PSNR = measurements_to_score(array_PSNR)
score = score_fm+score_drd+score_PSNR
print(score)

scores_dictionary = {}
for i, f in enumerate(filenames):
    scores_dictionary[f] = score[i]
df = df.append(scores_dictionary, ignore_index=True)
df['Metrics'] = ['MSE', 'BCE', 'BAC', 'F-measure', 'Precision', 'Recall', 'DRD', 'NRM', 'MPM', 'PSNR', 'Dice', 'mPSNR', 'Score']
if qualifier is None:
    df.to_excel('comparison_results.xlsx', index=False)
else:
    df.to_excel('comparison_results_{}.xlsx'.format(qualifier), index=False)
print(df)
