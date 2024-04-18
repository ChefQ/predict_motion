# %%
import pandas as pd
import argparse
import os
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Summarize the paired testset')
    parser.add_argument('-input', type=str, help='path to the input file')
    parser.add_argument('-output', type=str, help='path to the output file')
    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    if not os.path.exists(input_path):
        print(f'Input file {input_path} does not exist')
        exit(1)


    data = pd.read_csv(input_path, index_col=0)
    test = data.loc[data['data_type'] == 'test']
    test.to_csv(f'{output_path}.csv')


# %%


# %%



