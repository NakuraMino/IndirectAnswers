"""
Dataset class and utility functions to retrieve 
dataloader.

Useful links: 
- https://huggingface.co/transformers/model_doc/bert.html#berttokenizer
- https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
- https://github.com/google-research-datasets/boolean-questions
- https://github.com/google-research-datasets/circa
- https://cims.nyu.edu/~sbowman/multinli/
"""

import torch 
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import pandas as pd
import time
from transformers import BertTokenizer
import json

class CircaDataset(Dataset):
    """
    dataset class for the circa dataset 
    """

    def __init__(self, file_path, tokenizer=None, mode=None):
        """
        @param file_path: the path to a circa.tsv file
        @param tokenizer (default=None): an optional tokenizer to convert text 
                                         to tokens. If none is passed in, uses 
                                         the bert-base-uncased tokenizer.
        @param mode (default=None): sets whether we want to use the full input (question+response)
                                    or just question or just answer. To use only questions, use mode="q",
                                    and mode="a" for just answers 
        """
        self.file_path = file_path
        self.data = pd.read_csv(file_path, sep='\t')
        self.mode = mode
        if tokenizer == None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer
            
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx): 
        """
        @param idx: index into the dataset

        @return: dictionary mapping from th column titles to the value at each column + row(idx) pair
                dictionary is in the form:
                {
                    'judgements': string,
                    'question': string,
                    'goldstandard1': int,
                    'goldstandard2': int 
                }
                where goldstandard1/2 are labels, and question is the string input. 
        """
        judgement = self.data.loc[idx]["judgements"]
        label1 = self.labelToIdx(self.data.loc[idx]["goldstandard1"])
        label2 = self.labelToIdx(self.data.loc[idx]["goldstandard2"])    
        if self.mode == "a":
            question = str(self.data.loc[idx]["context"]) + \
                " [SEP] " +str(self.data.loc[idx]['answer-Y'])
        elif self.mode == "q": 
            question = str(self.data.loc[idx]["context"]) + \
                " [SEP] " + str(self.data.loc[idx]['question-X'])
        else:
            question = str(self.data.loc[idx]["context"]) + \
                " [SEP] " + str(self.data.loc[idx]['question-X']) + \
                " [SEP] " + str(self.data.loc[idx]['answer-Y'])
        print(question)
        indexed_tokens = self.tokenizer(question, return_tensors="pt")
        header_to_data = {
            "judgements": judgement,
        }
        header_to_data['question'] = question
        header_to_data["goldstandard1"] = label1
        header_to_data["goldstandard2"] = label2
        return header_to_data

    def labelToIdx(self, label):
        """
        maps a label from the circa dataset to a numerical value 
        
        @param label: the gold standard label
        @return: a value 0 - 8 which encodes the label
        """
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
            return 6
        elif label == "Probably no":
            return 7
        elif label == "I am not sure how X will interpret Y's answer":
            return 8
        else: # nan/NA case
            return 5

    def collate_fn(self, batch):
        """
        @param batch: list of datapoints (dictionaries), each from __getitem__
                     with length N 
        @returns: a dictionary with the following keys:
            { "judgements": list of strings of len == N,
              "goldstandard1": (N,) torch.LongTensor of labels
              "goldstandard2": (N,) torch.LongTensor of labels
              "input ids": (N,K) torch.floatTensor of tokens representing the input question 
              "token_type_ids": (N,K) torch.LongTensor of the token types 
              "attention_mask": (N,K) torch.LongTensor of the attention masks
            }
            where K is the maximum length input string.
        """
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

class BOOLQDataset(Dataset):
    """
    loads the BOOLQ dataset
    """

    def __init__(self, file_path, tokenizer=None):
        """
        @param file_path: the path to a train.jsonl file
        """
        with open(file_path, encoding='iso-8859-1') as f:
            # TODO: encoding is wrong
            self.json_lines = f.readlines()
        if tokenizer == None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer

    def __len__(self): 
        return len(self.json_lines)

    def __getitem__(self, idx):
        """
        @param idx: int index into the dataset
        @returns: dictionary of the format:
            {
                "question": string,
                "passage": string, 
                "answer": int where 1 is true and 0 is false,
                "title": string
            }
        """
        json_string = self.json_lines[idx]
        json_dict = json.loads(json_string)
        json_dict['answer'] = int(json_dict['answer'])
        return json_dict

    def collate_fn(self, batch):
        """
        @param batch: list of datapoints (dictionaries), each from __getitem__
                     with length N 
        @returns: a dictionary with the following keys:
            {
                'answer': (N,) torch.longTensor,
                "input ids": (N,K) torch.floatTensor of tokens representing the input question, 
                             passage, and title with separator tokens in between 
                "token_type_ids": (N,K) torch.LongTensor of the token types 
                "attention_mask": (N,K) torch.LongTensor of the attention masks
            }
            where K is the maximum length input string.
        """
        input_list = list()
        labels = torch.zeros((len(batch),)).long()
        idx = 0
        for data in batch: 
            labels[idx] = data['answer']
            input_string = data['question'] + " [SEP] " + data['passage'] + " [SEP] " + data['title']
            input_list.append(input_string)
            idx += 1
        batch_dict = dict()
        batch_dict['answer'] = labels
        batch_dict['strings'] = input_list
        indexed_tokens = self.tokenizer(input_list, padding=True, return_tensors="pt")
        for key in indexed_tokens:
            batch_dict[key] = indexed_tokens[key]
        return batch_dict

class MNLIDataset(Dataset):
    """
    loads the MNLI dataset
    """
    def __init__(self, file_path, tokenizer=None):
        """
        @param file_path: the path to a train.jsonl file
        """
        with open(file_path, encoding='iso-8859-1') as f:
            # TODO: encoding is wrong
            self.json_lines = f.readlines()
        if tokenizer == None:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        else:
            self.tokenizer = tokenizer

    def __len__(self): 
        return len(self.json_lines)

    def __getitem__(self, idx):
        """
        @param idx: int index into the dataset
        @returns: dictionary of the format:
        """
        json_string = self.json_lines[idx]
        json_dict = json.loads(json_string)
        json_dict["gold_label"] = self.labelToIdx(json_dict["gold_label"])
        return json_dict

    def collate_fn(self, batch):
        """
        @param batch: list of datapoints (dictionaries), each from __getitem__
                     with length N 
        @return: a dictionary with the following keys:
            {
                sent1input_ids: (N,K) torch.floatTensor of tokens representing the input sentence,
                sent2input_ids: (N,K) torch.floatTensor of tokens representing the input sentence,
                sent1token_type_ids: (N,K) torch.LongTensor of the token types,
                sent2token_type_ids: (N,K) torch.LongTensor of the token types,
                sentence1: the original input sentence,
                sentence2: the original second input sentence,
                sent1attention_mask: (N,K) torch.LongTensor of the attention masks,
                sent2attention_mask: (N,K) torch.LongTensor of the attention masks,
                gold_labels: (N,) torch.longTensor
            }
        """
        batch_dict = dict()
        sentence_list = list()
        labels = torch.zeros((len(batch),)).long()
        idx = 0
        for data in batch:
            sentence_list += [str(data["sentence1"]  + " [SEP] " + data["sentence2"])]
            labels[idx] = data["gold_label"]
            idx += 1
        sentence_tokenized = self.tokenizer(sentence_list, padding=True, return_tensors="pt")
        for key in sentence_tokenized:
            batch_dict["sentence_" + key] = sentence_tokenized[key]
        batch_dict["gold_labels"] = labels
        batch_dict["sentence"] = sentence_list
        return batch_dict

    def labelToIdx(self, label):
        """
        maps a label from the circa dataset to a numerical value 
        
        @param label: the gold label
        @return: a value 0 - 4 which encodes the label
        """
        if label == "entailment":
            return 0
        elif label == "neutral": 
            return 1
        elif label == "contradiction":
            return 2
        else:
            return 3

def getCircaDataloader(file_path, batch_size=16, num_workers=4, mode=None, shuffle=True, tokenizer=None):
    """
    creates a dataset and returns a dataloader 

    @param file_path: the path to a circa.tsv file
    @param batch_size (default=16): size of each batch
    @param num_workers (default=4): the number of workers
    @param shuffle (default=True): shuffle dataset or not (True or False value)
    @return: torch.utils.data.DataLoader object    
    """
    dataset = CircaDataset(file_path, mode=mode, tokenizer=tokenizer)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      collate_fn=dataset.collate_fn,
                      num_workers=num_workers)

def getBOOLQDataloader(file_path, batch_size=16, num_workers=4, shuffle=True, tokenizer=None):
    """
    creates a dataset and returns a dataloader 

    @param file_path: the path to a train.jsonl file
    @param batch_size (default=16): size of each batch
    @param num_workers (default=4): the number of workers
    @param shuffle (default=True): shuffle dataset or not (True or False value)
    @return: torch.utils.data.DataLoader object    
    """
    dataset = BOOLQDataset(file_path, tokenizer=tokenizer)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      collate_fn=dataset.collate_fn,
                      num_workers=num_workers)

def getMNLIDataloader(file_path, batch_size=16, num_workers=4, shuffle=True, tokenizer=None):
    """
    creates a dataset and returns a dataloader 

    @param file_path: the path to a train.jsonl file
    @param batch_size (default=16): size of each batch
    @param num_workers (default=4): the number of workers
    @param shuffle (default=True): shuffle dataset or not (True or False value)
    @return: torch.utils.data.DataLoader object    
    """
    dataset = MNLIDataset(file_path, tokenizer=tokenizer)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      collate_fn=dataset.collate_fn,
                      num_workers=num_workers)

if __name__ == "__main__":
    #testing MNLI dataset
    md = MNLIDataset("data/mnli/multinli_1.0_train.jsonl")
    dataloader = getMNLIDataloader("data/mnli/multinli_1.0_train.jsonl", batch_size=2, num_workers=1)
    dl_iter = iter(dataloader)
    batch = next(dl_iter)
    for key in batch:
        print(f"{key}: {type(batch[key])}")
    print(batch)
    # print(batch)

if False: #__name__ == "__main__":
    # testing BOOLQ dataset
    dataset = BOOLQDataset('./data/BoolQ/train.jsonl')
    print(dataset[0])
    dataloader = getBOOLQDataloader('./data/BoolQ/train.jsonl', batch_size=2, num_workers=1)
    dl_iter = iter(dataloader)
    batch = next(dl_iter)
    print(batch)


if False: #__name__ == "__main__":
    # testing circa dataset
    # some really simple testing 
    dataset = CircaDataset('./data/circa-data.tsv', mode='q')
    length = len(dataset)
    print(length)
    for key in dataset[0]:
        print(key)
    dataloader = getCircaDataloader('./data/circa-data.tsv', batch_size=2, num_workers=1, mode='a')
    dl_iter = iter(dataloader)
    batch = next(dl_iter)
    print(batch)
    for key in batch:
        print(key)
    print(type(batch))
    print(batch['judgements'])
    print(batch['goldstandard1'])
