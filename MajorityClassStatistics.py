"""
Calculates statistics for majority class, 
including F1 scores and accuracies 
"""

import sklearn.metrics as metrics
import dataloader 
import torch 
import statistics 

def getClassLabels(file_path):
    """
    gets majority class strict and reaxed labels for a dataset
    
    @return: tuple of (N,) torch.long vectors where first element is the 
             strict labels and the second element is the relaxed labels
    """
    dl = dataloader.getCircaDataloader(file_path)
    dl_iter = iter(dl)

    strict_labels = None
    relaxed_labels = None

    for batch in dl_iter: 
        
        # strict case
        strict_batch_labels = batch["goldstandard1"]
        if strict_labels is None: 
            strict_labels = strict_batch_labels
        else:      
            strict_labels = torch.cat([strict_labels, strict_batch_labels], dim=0)
        # relaxed case
        relaxed_batch_labels = batch["goldstandard2"]
        if relaxed_labels is None:
            relaxed_labels = relaxed_batch_labels
        else:
            relaxed_labels = torch.cat([relaxed_labels, relaxed_batch_labels], dim=0)
    return strict_labels, relaxed_labels

def printStatistics(pred_labels, y_labels, title="", p=True):
    f1_score = metrics.f1_score(y_labels, pred_labels, average=None)
    num_correct = torch.sum(pred_labels == y_labels)
    average = num_correct / y_labels.shape[0]
    if p:
        print(f"{title} accuracy: {average}")
        print(f"{title} f1 scores: {f1_score}")
        print()
    return f1_score, average

strict_test_labels, relaxed_test_labels = getClassLabels("./data/CIRCA/circa-data-test.tsv")
strict_dev_labels, relaxed_dev_labels = getClassLabels("./data/CIRCA/circa-data-dev.tsv")

shape = relaxed_dev_labels.shape
pred_labels = torch.zeros(shape)

printStatistics(strict_test_labels, pred_labels, "test strict")
printStatistics(strict_dev_labels, pred_labels, "dev strict")
printStatistics(relaxed_test_labels, pred_labels, "test relaxed")
printStatistics(relaxed_dev_labels, pred_labels, "dev relaxed")

strict_average_list = []
relaxed_average_list = []
for i in range(10):
    strict_test_labels, relaxed_test_labels = getClassLabels("./data/CIRCA/unmatched/scenario" + str(i) + "/circa-data-test.tsv")
    pred_labels = torch.zeros(strict_test_labels.shape)
    _, strict_average = printStatistics(pred_labels, strict_test_labels, p=False)
    _, relaxed_average = printStatistics(pred_labels, relaxed_test_labels, p=False)
    strict_average_list.append(strict_average.item())
    relaxed_average_list.append(relaxed_average.item())

print("strict unmatched statistics")
print(f"mean     : {sum(strict_average_list) / len(strict_average_list)}")
print(f"std. dev.: {statistics.pstdev(strict_average_list)}")
print(f"max      : {max(strict_average_list)}")
print(f"min      : {min(strict_average_list)}")
print()

print("relaxed unmatched statistics")
print(f"mean     : {sum(relaxed_average_list) / len(relaxed_average_list)}")
print(f"std. dev.: {statistics.pstdev(relaxed_average_list)}")
print(f"max      : {max(relaxed_average_list)}")
print(f"min      : {min(relaxed_average_list)}")
print()
