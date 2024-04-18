
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer
from typing import List, Set, Dict, Tuple
from torch.utils.data import Dataset
import torch.nn.functional as F
import ast
from joblib import dump, load
import torch.nn.functional as F
from torch.nn import (
    Sequential as Seq,
    Linear as Lin,
    ReLU,
    BatchNorm1d,
    AvgPool1d,
    Sigmoid,
    Conv1d,
)

class Data(Dataset):
    def __init__(self,df,feature = 'tfidf',support_pipe = './pipes/support-tfidf.joblib', opposition_pipe = './pipes/oppose-tfidf.joblib', both =False, both_pipe = './pipes/both-tfidf.joblib', getEmbeddings = None ):
        self.df = df    
        supports = self.df['support'].values
        oppositions = self.df['opposition'].values
        self.folder_id = self.df['folder_id'].values
        self.y = self.df['outcome'].values 
        # convert list of stings to list of lists of stings
        supports = list(map(lambda x: ast.literal_eval(x), supports))
        oppositions = list(map(lambda x: ast.literal_eval(x), oppositions))
        self.both = both
        if self.both:
            self.combined = list(map(lambda x,y: x+y, supports,oppositions))


        self.getEmbeddings = getEmbeddings
        
        if self.both == False:
            self.max_len_brief = max(self.findMaxLen(supports),self.findMaxLen(oppositions))
        else:
            self.max_len_brief = self.findMaxLen(self.combined)

        if feature == 'tfidf':
            if self.both == False:
                support_pipe = load(support_pipe)
                opposition_pipe = load(opposition_pipe)
                getSupport = lambda x: self.stringsToTfidfs(x,support_pipe)
                getOpposition = lambda x: self.stringsToTfidfs(x,opposition_pipe)


                self.supports = list(map( getSupport, supports))
                self.oppositions = list(map( getOpposition, oppositions))

            else:
                both_pipe = load(both_pipe)
                getTfidf= lambda x: self.stringsToTfidfs(x,both_pipe)
                self.combined = list(map( getTfidf, self.combined))

        elif feature == 'embedding':
            if self.both == False:
                self.supports: list = list(map(lambda x: self.stringsToEmbeddings(x), supports))
                self.oppositions: list = list(map(lambda x: self.stringsToEmbeddings(x), oppositions))
            else:
                self.combined: list = list(map(lambda x: self.stringsToEmbeddings(x), self.combined))

        
    def __len__(self):
        if self.both == False:
            return len(self.supports)
        else:
            return len(self.combined)
    
    def __getitem__(self, idx):
        y = 1.0 if self.y[idx] == 'grant' else 0.0

        if hasattr(self, 'combined') and self.both == True:
            return self.combined[idx] , y , self.folder_id[idx]
        else:
            return self.supports[idx] , self.oppositions[idx] , y , self.folder_id[idx]
        
    def findMaxLen(self,x):
        max_len = 0
        for i in range(len(x)):
            row = x[i]
            if len(row) > max_len:
                max_len = len(row)
        return max_len

    def stringsToTfidfs(self,briefs: List[str],pipe):
        tfidfs = torch.tensor(pipe.transform(briefs).toarray(),dtype=torch.float32)

        return self.padFeatures(tfidfs)
    
        # num_padding = self.max_len_brief - tfidfs.shape[0]
        # padding = nn.ConstantPad2d((0, 0, 0, num_padding), 0)
        # tfidfs = padding(tfidfs)
        # tfidfs = tfidfs.T
        # return tfidfs
    
    def stringsToEmbeddings(self,briefs: List[str]):
        if self.getEmbeddings == None:
            raise ValueError('No function to get embeddings')
        
        embeddings =  torch.tensor(self.getEmbeddings(briefs),dtype=torch.float32)
        return self.padFeatures(embeddings)
    
    def padFeatures(self,features: List[torch.tensor]):
        num_padding = self.max_len_brief - features.shape[0]
        padding = nn.ConstantPad2d((0, 0, 0, num_padding), 0)
        features = padding(features)
        features = features.T
        return features
   

