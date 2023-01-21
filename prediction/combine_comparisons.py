import pandas as pd
import argparse
import os
import sys

parser = argparse.ArgumentParser(description='Combine results from multiple validation sets')
parser.add_argument('--input-files', nargs='+', help='Input files', required=True)
args = parser.parse_args()

input_files = args.input_files

for file in input_files:
    if not os.path.exists(file) or not os.path.isfile(file):
        print('Bad input file')
        sys.exit(0)


def index_to_list(index):
    result = []
    for i in index:
        result.append(i[5:])
    return result


head = None
scores = []
for f in input_files:
    df = pd.read_excel(f)
    if head is None:
        head = index_to_list(df.columns)
    else:
        temp_head = index_to_list(df.columns)
        if head != temp_head:
            print('Incompatible files')
            sys.exit(0)
    df.columns = head
    scores.append(df.iloc[-1])

combined_df = pd.DataFrame(columns=head)
for score in scores:
    combined_df = pd.concat([combined_df, score])

print(combined_df)



'''
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
'''
