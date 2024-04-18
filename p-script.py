# %%
from transformers import pipeline, set_seed
from datasets import Dataset , load_dataset, DatasetDict
import pandas as pd
import json
import regex as re
import nltk
from nltk.tokenize import sent_tokenize
from transformers import AutoModelForSequenceClassification
import ast
import torch
from huggingface_hub import scan_cache_dir
import numpy as np
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel, AutoConfig
import ast
import torch.nn.functional as f
if torch.cuda.is_available():
    device = torch.device('cuda')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    device = torch.device('mps')
else:
    device = torch.device('cpu')
import os
from datasets import DatasetDict, concatenate_datasets
from sentence_transformers import SentenceTransformer
from joblib import dump, load
import argparse
import joblib
from datasets import Features , ClassLabel, Value, Sequence
from utility import Data

#os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

#watch -n0.1 nvidia-smi



from per_motion_prediction.deepsetmodel import *

# %%
### Load model


# %%
def getIdsType(brief_type):
    def getIds(briefs):

        briefs = briefs[brief_type]
        briefs_ids = []
        briefs = ast.literal_eval(briefs)
        for brief in briefs:
            iD = tokenizer(brief, max_length = max_input_size, padding='max_length', truncation= True ,return_tensors="pt") #.to(device).input_ids 
            briefs_ids.append(iD)
        return { f"ids_{brief_type}": briefs_ids }

    return getIds


def meanEmbeddings(briefs):

    briefs = ast.literal_eval(briefs)
    # Place model inputs on the GPU
    embeddings = []
    for brief in briefs:
        argument = tokenizer(brief, max_length = max_input_size , padding="max_length" ,truncation= True ,return_tensors="pt").to(device) 
        # Extract last hidden states
        model.eval()
        with torch.no_grad():
            argument = argument.to(device)
            output = model(**argument)
            argument = argument.to(device)
            last_hidden_state = output.last_hidden_state


        last_hidden_state = last_hidden_state.reshape(( max_input_size, config.hidden_size))
        mask = argument['attention_mask'].to(device).bool()
        mask = mask.reshape(max_input_size)
        mask = mask.nonzero().squeeze()
        hidden_states = torch.index_select(last_hidden_state, 0, mask)
        
        #print(last_hidden_state)
        inputs = get_mean_embedding(hidden_states.cpu().to(torch.float64).numpy()).tolist()
        del last_hidden_state
        del output
        del mask
        del hidden_states
        del argument
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
                
        embeddings.append(inputs)
    # Return vector for [CLS] token
    return embeddings

def summerizeEmbeddings(brief):
    embedding = sentence_model.encode(brief).tolist()
    return embedding


def tokenize_function(briefs):
    return tokenizer(briefs["prompt"], padding="max_length", truncation=True)


def get_mean_embedding(embedding):
    return np.mean(embedding, axis=0)

# There is an issue here. I Deepset model's input is maxed at 10 breifs.
# If the new Dataset has more than 10 briefs per motion, then the model will not be able to handle it.
        
def deepSetPredictions(loader, model_state_path ,input_size, max_len_brief, datatype):

################## Predifined model ############################
    hidden1 = int(input_size /5)
    hidden2 = int(hidden1 / 4)
    hidden3 = int(hidden2 / 3)
    classify1 = int(hidden3 /2)
    model = DeepSets(input_size, max_len_brief , hidden1, hidden2, hidden3, classify1).to(device)
    model.load_state_dict(torch.load(model_state_path))

########################################################################
    csv = {'folder':[],'prediction':[], 'score':[], 'truth':[]}
    model.eval()
    with torch.no_grad():

        for data in loader:
                
            if datatype != "both":
                supports, oppositions, y , folder_id = data
                supports = supports.to(device)
                oppositions = oppositions.to(device)
            else:
                combined, y , folder_id = data
                combined = combined.to(device)

            y = y.float()
            y = y.reshape(-1,1)
            y = y.to(device)

            if datatype == 'support':
                outputs= model(supports)
            elif datatype == 'opposition':
                outputs= model(oppositions)
            elif datatype == 'both':
                outputs= model(combined)

            predictions = (outputs > 0.5)
            csv['folder'].extend(folder_id)
            

            csv['prediction'].extend(predictions.cpu().numpy().flatten())
            

            csv['score'].extend(outputs.cpu().numpy().flatten())
            
            csv['truth'].extend(y.cpu().numpy().flatten())

    csv = pd.DataFrame(csv)

    csv["folder"] = csv["folder"].astype(int)
    csv["prediction"] = ["grant" if x == 1 else "deny" for x in csv["prediction"]]
    csv["truth"] = ["grant" if x == 1 else "deny" for x in csv["truth"]]

    del model
    torch.cuda.empty_cache()

    return csv



def test_metrics(model, dataloader):
    csv = {'brief':[],'predict':[], 'score':[], 'truth':[]}

    model.eval()
    for batch in dataloader:
        briefs = batch['file_name']
        inputs = {k: v.to(device) for k, v in batch.items() if k != "file_name"}
        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        propabilities = f.softmax(logits, dim=-1)
        predictions = torch.argmax(logits, dim=-1)

        csv['brief'].extend(briefs)
        labels = lambda x: "grant" if x == 1 else "deny"
        predict = list(map(labels, predictions))

        # return the probability of grant always.
        # and check this for other models

        csv['predict'].extend(predict) 
        # index 1 refers to the probability of grant
        csv['score'].extend(propabilities[:,1].cpu().numpy())
        csv['truth'].extend(list(map(labels, inputs["labels"].cpu().numpy())))

    return pd.DataFrame(csv)

def checkCorrectFormat(df):
    if 'support' not in df.columns and "opposition" not in df.columns:
        raise ValueError("The dataframe must have a column named 'support' and 'opposition'. With each row representing a set of briefs for a motion")


def decision2label(decision):
    if  "grant" in decision:
        return 1
    elif "deny" in decision:
        return 0
    else:
        #Note remember to change
        #print(f"error occured with decision: {decision} ",)
        return 0
        exit("Invalid decision")

#Todo: Remove the scientific notation from the csv file
        
#write script that converts folder to input csv files
        
# make preidcion files with only test set
        
# checkout the bottom half of the eval file after sorting AUC
                
# Descripe

# model_name = "KNN"
# feature = 'tfidf'
# #for model_name in ["RFT", "SGD"]: #["KNN","LinearSVC", "Logistic", "RFT", "SGD"]:
# data = 'dataset/testset.csv'
if __name__ == "__main__": #True:
    model_types = ["KNN","LinearSVC", "Logistic", "RFT", "SGD"]
    parser = argparse.ArgumentParser(description='End to end pipeline from briefs to predictions')
    parser.add_argument('--model_name', default='RFT', help=f'There are {len(model_types)} models to choose from: {model_types}')
    parser.add_argument('--data',  default= 'embeddings.csv')
    parser.add_argument('--feature', default=None, help='There are two features to choose from: sentence_embeddings and tfidf')

    parser.add_argument('--combine', action='store_true', help='Combine the briefs for predictions')

    arg = parser.parse_args()

    testset = pd.read_csv(arg.data, index_col=0)# pd.read_csv(data, index_col=0)#

    if not arg.combine:
        if arg.feature in ["sentence_embeddings", "tfidf"]:
            testset["completion"] = list(map(lambda x : x.strip() ,testset["completion"].to_list()))
            

            summerize = True

            if arg.feature == 'sentence_embeddings':#if arg.feature == 'sentence_embeddings':
                if 'embeddings' not in testset.columns:
                    if not summerize:
                        model  = "mistralai/Mistral-7B-v0.1"
                        config = AutoConfig.from_pretrained(model)
                        max_input_size =  config.max_position_embeddings  
                        tokenizer = AutoTokenizer.from_pretrained(model, device=device, )

                        tokenizer.pad_token = tokenizer.eos_token
                        model = AutoModel.from_pretrained(model, torch_dtype= torch.bfloat16 ).to(device)  #,device_map = "auto"
                        #parallel_model = torch.nn.DataParallel(model)
                    else:
                        sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)

                    testset['feature'] = ""
                    testset['feature'] = testset['prompt'].map(summerizeEmbeddings)
                    testset.to_csv('embeddings.csv')


                else:
                    #get model for predictions
                    turn2list = lambda x: ast.literal_eval(x)
                    testset['feature'] = testset['embeddings'].map(turn2list)


            elif arg.feature == 'tfidf':# elif arg.feature == 'tfidf':
                print("Loading tfidf pipes")
                support_pipe = joblib.load('pipes/support-tfidf.joblib')
                opposition_pipe = joblib.load('pipes/oppose-tfidf.joblib')

                testset['feature'] = ""

                # sparse_matrix = support_pipe.transform(testset["prompt"].loc[testset["brief_type"]=="support"]).toarray()

                # testset['feature'].loc[testset["brief_type"]=="support"] =   sparse_matrix.tolist()

                # sparse_matrix = opposition_pipe.transform(testset["prompt"].loc[testset["brief_type"]=="opposition"]).toarray()

                # testset['feature'].loc[testset["brief_type"]=="opposition"] =   sparse_matrix.tolist()


                sparse_matrix = support_pipe.transform(testset["prompt"].loc[testset["brief_type"]=="support"]).toarray().tolist()

                testset.loc[testset["brief_type"]=="support" , 'feature'] = pd.Series( sparse_matrix , index = testset.loc[testset["brief_type"]=="support"].index)

                sparse_matrix = opposition_pipe.transform(testset["prompt"].loc[testset["brief_type"]=="opposition"]).toarray().tolist()

                testset.loc[testset["brief_type"]=="opposition", 'feature'] =  pd.Series( sparse_matrix , index = testset.loc[testset["brief_type"]=="opposition"].index)
                        

            x_support = np.array(testset["feature"].loc[(testset["brief_type"]=="support") & (testset["data_type"]=="test") ].to_list())  #  np.array(testset['file_path'].to_list())
            x_opposition = np.array(testset["feature"].loc[(testset["brief_type"]=="opposition") & (testset["data_type"]=="test") ].to_list())  #  np.array(testset['file_path'].to_list())
                

            #get model for predictions

            support_model_path = f'models/{arg.model_name}-support-{arg.feature}.pkl'
            opposition_model_path = f'models/{arg.model_name}-opposition-{arg.feature}.pkl'

            clfs = {"sup" : load(support_model_path)  , "opp" : load(opposition_model_path) }

            if hasattr(clfs["sup"], 'predict_proba') and callable(getattr(clfs["sup"], 'predict_proba')):
                scores_support = clfs['sup'].predict_proba(x_support)
                scores_opposition = clfs['opp'].predict_proba(x_opposition)
            else:
                scores_support = clfs['sup'].decision_function(x_support)
                scores_opposition = clfs['opp'].decision_function(x_opposition)

            prediction_opposition = clfs['opp'].predict(x_opposition)
            prediction_support = clfs['sup'].predict(x_support)

            testset.rename(columns={"file_name": "brief","completion": "truth"} , inplace = True  )

            support = testset.loc[(testset["brief_type"]=="support") & (testset["data_type"]=="test") ].copy()
            opposition = testset.loc[(testset["brief_type"]=="opposition") & (testset["data_type"]=="test") ].copy()

            support.drop(['data_type','prompt' , 'brief_type','file_path','feature'], axis=1, inplace=True)
            opposition.drop(['data_type', 'prompt' , 'brief_type','file_path','feature'], axis=1, inplace=True)

            support['predict'] = ""
            opposition['predict'] = ""

            support['predict'] = prediction_support
            opposition['predict'] = prediction_opposition

            support['score'] = ""
            opposition['score'] = ""

            support['score'] = list(map( np.max ,scores_support.tolist()))
            opposition['score']= list(map(  np.max ,scores_opposition.tolist() ))

            support = support[["brief","predict","score","truth"]]
            opposition = opposition[["brief","predict","score","truth"]]

            support.to_csv(f'predictions/{arg.model_name}-{arg.feature}-supppredictions.csv' , index = False)
            opposition.to_csv(f'predictions/{arg.model_name}-{arg.feature}-oppopredictions.csv', index = False)

        elif arg.model_name == "LLM":

            testset['labels'] = testset['completion'].apply(decision2label)

            support_test = testset.loc[testset['brief_type'] == "support"]
            opposition_test = testset.loc[testset['brief_type'] == "opposition"]

            model_type = "bert"

            bert_checkpoint = "bert-base-uncased"

            

            # can change the argument 
             
            support_test = Dataset.from_pandas(support_test, preserve_index=False ) #, features= features )
            opposition_test = Dataset.from_pandas(opposition_test, preserve_index=False ) #, features= features )

            dataset = DatasetDict()
            dataset['support'] = support_test
            dataset['opposition'] = opposition_test
            tokenizer = AutoTokenizer.from_pretrained(bert_checkpoint)

            tokenized_datasets = dataset.map(tokenize_function, batched=True)
            tokenized_datasets = tokenized_datasets.remove_columns(["completion","prompt","brief_type","data_type", "file_path", ]) # "file_name"])
            tokenized_datasets.set_format("torch")

            support_loader = DataLoader(tokenized_datasets["support"], batch_size=16, shuffle=False) 
            opposition_loader = DataLoader(tokenized_datasets["opposition"], batch_size=16, shuffle=False)

            model = AutoModelForSequenceClassification.from_pretrained("models/LLM-bert-support-test", num_labels=2).to(device)
            model.eval()

            support_csv = test_metrics(model, support_loader)

            support_csv.to_csv(f'predictions/{arg.model_name}-{bert_checkpoint}-supppredictions.csv', index = False)
        
            del model
            torch.cuda.empty_cache()

            model = AutoModelForSequenceClassification.from_pretrained("models/LLM-bert-opposition-test").to(device)
            model.eval()

            opposition_csv = test_metrics(model, opposition_loader)

            opposition_csv.to_csv(f'predictions/{arg.model_name}-{bert_checkpoint}-oppopredictions.csv', index = False)

            del model
            torch.cuda.empty_cache()

    else:
        checkCorrectFormat(testset)
        
        if arg.feature == 'sentence_embeddings':
            sentence_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').to(device)
            
            both_data = Data(testset, feature="embedding" ,both=True, getEmbeddings = sentence_model.encode)
            both_loader = torch.utils.data.DataLoader(both_data, batch_size=32, shuffle=False)

            model_state_path = 'models/DeepSets_both_embedding.pth'
            input_size = both_data.combined[0].shape[0]
            max_len_brief = both_data.max_len_brief
            both_csv = deepSetPredictions(both_loader, model_state_path ,input_size, max_len_brief, "both")
            both_csv.to_csv(f'predictions/DeepSet-{arg.feature}-bothpredictions.csv', index = False)
        
            data = Data(testset, feature="embedding" ,both=False, getEmbeddings = sentence_model.encode)
            loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False)

            model_state_path = 'models/DeepSets_support_embedding.pth'
            input_size = data.supports[0].shape[0]
            max_len_brief = data.max_len_brief
            supportcsv = deepSetPredictions(loader, model_state_path ,input_size, max_len_brief, "support")
            supportcsv.to_csv(f'predictions/DeepSet-{arg.feature}-supppredictions.csv', index = False)

            model_state_path = 'models/DeepSets_opposition_embedding.pth'
            input_size = data.oppositions[0].shape[0]
            max_len_brief = data.max_len_brief
            oppositioncsv = deepSetPredictions(loader, model_state_path ,input_size, max_len_brief, "opposition")
            oppositioncsv.to_csv(f'predictions/DeepSet-{arg.feature}-oppopredictions.csv', index = False)

        elif arg.feature == 'tfidf':
            print("Loading tfidf pipes")

            both_data = Data(testset, feature="tfidf" ,both=True, getEmbeddings = None)
            both_loader = torch.utils.data.DataLoader(both_data, batch_size=32, shuffle=False)

            model_state_path = 'models/DeepSets_both_tfidf.pth'
            input_size = both_data.combined[0].shape[0]
            max_len_brief = both_data.max_len_brief
            both_csv = deepSetPredictions(both_loader, model_state_path ,input_size, max_len_brief, "both")
            both_csv.to_csv(f'predictions/DeepSet-{arg.feature}-bothpredictions.csv', index = False)

            data = Data(testset, feature="tfidf" ,both=False, getEmbeddings = None)
            loader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=False)

            model_state_path = 'models/DeepSets_support_tfidf.pth'
            input_size = data.supports[0].shape[0]
            max_len_brief = data.max_len_brief
            supportcsv = deepSetPredictions(loader, model_state_path ,input_size, max_len_brief, "support")
            supportcsv.to_csv(f'predictions/DeepSet-{arg.feature}-supppredictions.csv', index = False)

            model_state_path = 'models/DeepSets_opposition_tfidf.pth'
            input_size = data.oppositions[0].shape[0]
            max_len_brief = data.max_len_brief
            oppositioncsv = deepSetPredictions(loader, model_state_path ,input_size, max_len_brief, "opposition")
            oppositioncsv.to_csv(f'predictions/DeepSet-{arg.feature}-oppopredictions.csv', index = False)

            



        else:
            raise ValueError("The feature must be either sentence_embeddings or tfidf")
        
        # do this tomorrow
