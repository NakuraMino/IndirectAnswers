import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def train(model, dataloader, optimizer, device, pretrained_bool):
    for texts, labels in tqdm(dataloader):
        texts, labels = texts.to(device), labels.to(device)
        output = model(texts)
        loss = F.cross_entropy(output, labels)
        model.zero_grad()
        loss.backward()
        optimizer.step()


def evaluate(model, dataloader, device):
    count = correct = 0.0
    f1 = None
    actual = []
    pred = []
    with torch.no_grad():
        for texts, labels in tqdm(dataloader):
            texts, labels = texts.to(device), labels.to(device)
            # shape: (batch_size, n_labels)
            output = model(texts)
            # shape: (batch_size,)
            predicted = output.argmax(dim=-1)
            count += len(predicted)
            correct += (predicted == labels).sum().item()
            pred.extend([val.item() for val in predicted])
            actual.extend([val.item() for val in labels])
    accuracy = accuracy_score(actual, pred)
    print(f"Accuracy: {accuracy}")
    return accuracy

