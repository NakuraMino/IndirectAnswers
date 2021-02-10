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
from transformers import BertTokenizer
# from transformers.

class CircaDataset(Dataset):
    """
    dataset class for the circa dataset 
    """

    def __init__(self, file_path):
        """
        @param file_path: the path to a circa.tsv file
        """
        self.file_path = file_path
        self.data = pd.read_csv(file_path, sep='\t')
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        print(type(self.tokenizer))
    
    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        header_to_data = dict()
        for header in self.data.columns:
            if header == "id":
                data = self.data.loc[idx][header]                                
            elif header == "judgements" or header == "goldstandard1" or header == "goldstandard2":
                # data labels
                data = self.data.loc[idx][header]
            else: 
                # context, question-X, canquestion-X, answer-Y
                text = str(self.data.loc[idx][header]) 
                tokenized_text = self.tokenizer.tokenize(text)
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
                data = indexed_tokens
            header_to_data[header] = data
        return header_to_data


def getCircaDataloader(file_path, batch_size=16, num_workers=4, shuffle=True):
    dataset = CircaDataset(file_path)
    return DataLoader(dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers)

if __name__ == "__main__":
    dataset = CircaDataset('./data/circa-data.tsv')
    length = len(dataset)
    print(length)

    print(dataset[0])

    dataloader = getCircaDataloader('./data/circa-data.tsv', num_workers=1)
    dl_iter = iter(dataloader)
    print(next(dl_iter))