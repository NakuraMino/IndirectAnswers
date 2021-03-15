"""
Calculates statistics for majority class, 
including F1 scores and accuracies 
"""

import sklearn.metrics as metrics
import dataloader 
import torch 

dl = dataloader.getCircaDataloader("./data/CIRCA/circa-data-test.tsv")
dl_iter = iter(dl)

strict_pred_labels = None
relaxed_pred_labels = None

for batch in dl_iter: 
    
    # strict case
    strict_labels = batch["goldstandard1"]
    if strict_pred_labels is None: 
        strict_pred_labels = strict_labels
    else:      
        strict_pred_labels = torch.cat([strict_pred_labels, strict_labels], dim=0)
    # relaxed case
    relaxed_labels = batch["goldstandard2"]
    if relaxed_pred_labels is None:
        relaxed_pred_labels = relaxed_labels
    else:
        relaxed_pred_labels = torch.cat([relaxed_pred_labels, relaxed_labels], dim=0)

shape = relaxed_pred_labels.shape
pred_labels = torch.zeros(shape)
strict_f1_score = metrics.f1_score(strict_pred_labels, pred_labels, average=None)
relaxed_f1_score = metrics.f1_score(relaxed_pred_labels, pred_labels, average=None)

print("strict f1 scores: ", strict_f1_score)
print("relaxed f1_scores: ", relaxed_f1_score)
        

