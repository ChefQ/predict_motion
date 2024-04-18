
import pandas as pd
import os
import argparse
import re
from transformers import pipeline
import torch
from tqdm import tqdm

SUFFIX = "\n Judgment -->"#'\n\n###\n\n'
if __name__ == "__main__": 

        parser = argparse.ArgumentParser(description='Accepts a folder path and converts all the files in the folder to a single csv file')
        parser.add_argument('--folder_path', type=str, help='The path to the folder containing the files')
        parser.add_argument('--csv_type', type=str, help='The type of csv paired or unpaired or return both')
        parser.add_argument('--summarize', action='store_true', help='If true, the program will return summaries of the briefs')
        args = parser.parse_args()

        folder_path = args.folder_path
        csv_type = args.csv_type

        if not os.path.exists(folder_path):
            print('The folder path does not exist')
            exit()
        
        if csv_type not in ['paired', 'unpaired', "both"]:
            print('The csv type is not valid')
            exit()

        if args.summarize:
            summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device= torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))

        if csv_type == 'unpaired' or csv_type == 'both':

            index_train = pd.read_csv(f'{folder_path}/index-train.csv', header=None)
            index_test = pd.read_csv(f'{folder_path}/index-test.csv', header=None)

            tags = {}
            for row in index_train.iterrows():
                index_name = re.match(r'.*/(\d+)/.*',row[1][0])[1]
                file_name = re.match(r'.*/.*/.*/(.*[.]txt)',row[1][0])[1]
                if index_name in tags.keys():
                    tags[index_name].append((row[1][1], row[1][2], 'train', file_name))
                else:
                    tags[index_name] = [(row[1][1], row[1][2], 'train', file_name)]
                
            for row in index_test.iterrows():
                index_name = re.match(r'.*/(\d+)/.*',row[1][0])[1]
                file_name = re.match(r'.*/.*/.*/(.*[.]txt)',row[1][0])[1]
                if index_name in tags.keys():
                    tags[index_name].append((row[1][1], row[1][2], 'test', file_name))
                else:
                    tags[index_name] = [ (row[1][1], row[1][2], 'test', file_name) ]

            temp_pandas = {'prompt': [], 'completion': [], 'brief_type': [], 'data_type': [] , 'file_path': [] , 'file_name' : []}
            # add tqdm to show progress

            for filename in tqdm(os.listdir(f'{folder_path}/files')):
                if filename in tags.keys():
                    if filename == '909':
                        print("here")
                        #pdb.set_trace()
                    for tag in tags[filename]:
                        brief_type = tag[0]
                        outcome = tag[1]
                        data_type = tag[2]
                        brief = tag[3]
                        for file in os.listdir(f'{folder_path}/files/'+filename):
                            if file == brief_type:
                                
                                with open(f'{folder_path}/files/'+filename+'/'+file+'/'+brief, 'r') as f:
                                    prompt = f.read()
                 
                                    if args.summarize:
                                        prompt = summarizer(prompt, max_length=200, min_length=154, do_sample=True, truncation = True)[0]['summary_text']

                                    prompt = prompt + SUFFIX
                                    completion = ' '+ outcome 
                                    
                                    temp_pandas['prompt'].append(prompt)
                                    temp_pandas['completion'].append(completion)
                                    temp_pandas['brief_type'].append(brief_type)
                                    temp_pandas['data_type'].append(data_type)
                                    temp_pandas['file_name'].append(brief)
                                    temp_pandas['file_path'].append(filename)
                                    
                                f.close()
                            

            #convert dictionary to pandas dataframe
            df = pd.DataFrame.from_dict(temp_pandas)
            if args.summarize:
                df.to_csv('summarized_testset.csv')
            else:
                df.to_csv('testset.csv')

        if csv_type == 'paired' or csv_type == 'both':
            data = {
                'support': [],
                'support_file': [],
                'opposition': [],
                'opposition_file': [],
                'outcome': [],
                'folder_id': [],
                'case_path': [],
                'data_type': []
            }

            training_ids = []
            testing_ids = []

            with open(f'{folder_path}/index-train.csv', 'r') as f:
                for line in f:
                    result = re.match(r'.*/(\d+)/.*',line)
                    training_ids.append(result.group(1))
                f.close()

            with open(f'{folder_path}/index-test.csv', 'r') as f:
                for line in f:
                    result = re.match(r'.*/(\d+)/.*',line)
                    testing_ids.append(result.group(1))
                f.close()


            for filename in tqdm(os.listdir(f'{folder_path}/files')):
                data['folder_id'].append(filename)
                if filename in training_ids:
                    data['data_type'].append('train')
                elif filename in testing_ids:
                    data['data_type'].append('test')
                else:
                    data['data_type'].append('none')
                with open(f'{folder_path}/files/'+filename+'/outcome', 'r') as f:
                    outcome = f.read()
                    result = re.match(r'\d.+(/mnt.+),(.+)', outcome) 
                    data['case_path'].append(result.group(1))
                    outcome = result.group(2)
                    data['outcome'].append(outcome)
                    
                    f.close()

                for file in os.listdir(f'{folder_path}/files/'+filename):
                    
                    if file in ['support', 'opposition' ]:
                        briefs = os.listdir(f'{folder_path}/files/'+filename+'/'+file+'/')

                        if len(briefs) > 1:
                            specific_briefs = []
                            brief_name = []
                            for brief in briefs:
                                with open(f'{folder_path}/files/'+filename+'/'+file+'/'+ brief, 'r') as f:                                     
                                    content = f.read()
                                    if args.summarize:
                                        content = summarizer(content,  max_length=200, min_length=154, do_sample=True, truncation = True)[0]['summary_text']
                                    specific_briefs.append(content)
                                    brief_name.append(brief)
                                f.close()
                            if file == 'opposition':
                                data['opposition'].append(specific_briefs)
                                data['opposition_file'].append(brief_name)
                            elif file == 'support':
                                data['support'].append(specific_briefs)
                                data['support_file'].append(brief_name)
                        else:
                            with open(f'{folder_path}/files/'+filename+'/'+file+'/'+briefs[0], 'r') as f:
                                content = f.read()
                                if args.summarize:
                                    content = summarizer(content,  max_length=200, min_length=154, do_sample=True, truncation = True)[0]['summary_text']

                                if file == 'opposition':
                                    data['opposition'].append([content])
                                    data['opposition_file'].append([briefs[0]])
                                elif file == 'support':
                                    data['support'].append([content])
                                    data['support_file'].append([briefs[0]])
                            f.close()


            df = pd.DataFrame.from_dict(data)
            df.head(10)
            if args.summarize:
                df.to_csv('summarized_paired_testset.csv')
            else:
                df.to_csv('paired_testset.csv')