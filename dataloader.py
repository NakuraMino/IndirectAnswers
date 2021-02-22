"""
Dataset class and utility functions to retrieve 
dataloader.

Useful links: 
- https://huggingface.co/transformers/model_doc/bert.html#berttokenizer
- https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
"""

import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
import time
from transformers import BertTokenizer

class CircaDataset(Dataset):
    """
    dataset class for the circa dataset 
    """

    def __init__(self, file_path, tokenizer=None, use_tokenizer=False):
        """
        @param file_path: the path to a circa.tsv file
        @param tokenizer (default=None): an optional tokenizer to convert text 
                                         to tokens. If none is passed in, uses 
                                         the bert-base-uncased tokenizer. 
        """
        self.file_path = file_path
        self.data = pd.read_csv(file_path, sep='\t')
        self.use_tokenizer = use_tokenizer
        if tokenizer == None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer
        time.sleep(2)
            
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx): 
        """
        @param idx: index into the dataset

        @return: dictionary mapping from th column titles to the value at each column + row(idx) pair
        """
        judgement = self.data.loc[idx]["judgements"]
        label1 = self.labelToIdx(self.data.loc[idx]["goldstandard1"])
        label2 = self.labelToIdx(self.data.loc[idx]["goldstandard2"])
        question = str(self.data.loc[idx]["context"]) + \
                   "[SEP] " + str(self.data.loc[idx]['question-X']) + \
                   "[SEP] " + str(self.data.loc[idx]['answer-Y'])
        indexed_tokens = self.tokenizer(question, return_tensors="pt")
        header_to_data = {
            "judgements": judgement,
        }
        header_to_data['question'] = question
        if self.use_tokenizer:
            for key in label1:
                header_to_data["goldstandard1_" + key] = label1[key]
                header_to_data["goldstandard2_" + key] = label2[key]
        else:
            header_to_data["goldstandard1"] = label1
            header_to_data["goldstandard2"] = label2
        return header_to_data

    def labelToIdx(self, label):
        """
        maps a label from the circa dataset to a numerical value 
        
        @param label: the gold standard label
        @return: a value 1 - 8 which encodes the label
        """
        if self.use_tokenizer: 
            return self.tokenizer(label, return_tensors="pt")

        if label == 'Yes':
            return 0
        elif label == 'No':
            return 1
        elif label == 'Yes, subject to some conditions':
            return 2
        elif label == "In the middle, neither yes nor no":
            return 3
        elif label == "Other":
            return 4
        elif label == 'Probably yes / sometimes yes':
            return 5
        elif label == "Probably no":
            return 6
        elif label == "I am not sure how X will interpret Y's answer":
            return 7
        else: # nan/NA case
            return 8

    def collate_fn(self, batch):
        batch_dict = dict()
        batch_size = len(batch)
        # judgement
        batch_dict['judgements'] = list()    
        for data in batch:
            batch_dict['judgements'].append(data['judgements'])
        # label
        batch_dict["goldstandard1"] = torch.zeros((len(batch),)).long()
        batch_dict["goldstandard2"] = torch.zeros((len(batch),)).long()
        idx = 0
        for data in batch:
            batch_dict["goldstandard1"][idx] = data['goldstandard1']
            batch_dict["goldstandard2"][idx] = data['goldstandard2']
            idx += 1
        # input strings
        questions_list = list()
        for data in batch:
            questions_list.append(data["question"])
        indexed_tokens = self.tokenizer(questions_list, padding=True, return_tensors="pt")
        for key in indexed_tokens:
            batch_dict[key] = indexed_tokens[key]
        return batch_dict

def getCircaDataloader(file_path, batch_size=16, num_workers=4, shuffle=True, tokenizer=None, use_tokenizer=False):
    """
    creates a dataset and returns a dataloader 

    @param file_path: the path to a circa.tsv file
    @param batch_size (default=16): size of each batch
    @param num_workers (default=4): the number of workers
    @param shuffle (default=True): shuffle dataset or not (True or False value)
    @return: torch.utils.data.DataLoader object    
    """
    dataset = CircaDataset(file_path, tokenizer=tokenizer, use_tokenizer=use_tokenizer)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      collate_fn=dataset.collate_fn,
                      num_workers=num_workers)

if __name__ == "__main__":
    # some really simple testing 
    dataset = CircaDataset('./data/circa-data.tsv')
    length = len(dataset)
    print(length)
    for key in dataset[0]:
        print(key)
    dataloader = getCircaDataloader('./data/circa-data.tsv', batch_size=2, num_workers=1, use_tokenizer=False)
    dl_iter = iter(dataloader)
    batch = next(dl_iter)
    print(batch)
    print(type(batch))
    print(batch['judgements'])
    print(batch['goldstandard1'])
