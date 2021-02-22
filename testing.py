import sys
import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import BertForSequenceClassification
from sklearn.metrics import accuracy_score

from datasets import load_metric
metric = load_metric("accuracy")


def eval(model, dataloader, device):
    count = correct = 0.0
    actual = []
    pred = []
    with torch.no_grad():
        for texts, labels in tqdm(dataloader):
            texts, labels = texts.to(device), labels.to(device)
            # shape: (batch_size, n_labels)
            output = model(texts)
            # shape: (batch_size,)
            predicted = output.argmax(dim=-1)
            pred.extend([val.item() for val in predicted])
            actual.extend([val.item() for val in labels])
    accuracy = accuracy_score(actual, pred)
    print(f"Accuracy: {accuracy}")
    return accuracy

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python testing.py modelPath dataloader")
    path = sys.argv[1]
    dataloader = sys.argv[2]
    model = BertForSequenceClassification()
    model.load_state_dict(torch.load(path))

    eval(model, dataloader, device)