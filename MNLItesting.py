from argparse import ArgumentParser
import torch
from tqdm import tqdm
from transformers import BertForSequenceClassification, BertTokenizer, BertConfig
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np
import os.path

import dataloader


def evaluate(args, model, tokenizer, device):
    print("Evaluating model on test data...")
    
    # Loads a dataset
    test_dataloader = None
    if args.dataset_type == 'CIRCA':
        test_dataloader = dataloader.getCircaDataloader(args.test_data, batch_size=args.batch_size, num_workers=4, tokenizer=tokenizer, mode=args.dataset_mode)
    elif args.dataset_type == 'BOOLQ':
        test_dataloader = dataloader.getBOOLQDataloader(args.test_data, batch_size=args.batch_size, num_workers=4, tokenizer=tokenizer)
    elif args.dataset_type == 'MNLI':
        test_dataloader = dataloader.getMNLIDataloader(args.test_data, batch_size=args.batch_size, num_workers=4, tokenizer=tokenizer)
    else:
        test_dataloader = dataloader.getDISDataloader(args.test_data, batch_size=args.batch_size, num_workers=4, tokenizer=tokenizer)
    
    model.to(device)
    model.eval()

    actual = []
    pred = []
    for batch in tqdm(test_dataloader, desc="Checking model accuracy..."):
        with torch.no_grad():
            if args.dataset_type == 'CIRCA':
                if args.num_labels == 4:  # Strict case
                    # == 4 is a hack to test MNLI baseline model
                    input_ids, atten, labels, token_type_id = batch['input_ids'], batch['attention_mask'], batch['goldstandard1'], batch['token_type_ids']
                else:  # Relaxed case
                    input_ids, atten, labels, token_type_id = batch['input_ids'], batch['attention_mask'], batch['goldstandard2'], batch['token_type_ids']
            elif args.dataset_type == 'BOOLQ':
                input_ids, atten, labels, token_type_id = batch['input_ids'], batch['attention_mask'], batch['answer'], batch['token_type_ids']
            elif args.dataset_type == 'MNLI':
                input_ids, atten, labels, token_type_id = batch['sentence_input_ids'], batch['sentence_attention_mask'], batch['gold_labels'], batch['sentence_token_type_ids']
            else:
                input_ids, atten, labels, token_type_id = batch['input_ids'], batch['attention_mask'], batch['labels'], batch['token_type_ids']

            input_ids = input_ids.to(device)
            atten = atten.to(device)
            batch_size = input_ids.shape[0]
            labels = labels.to(device).squeeze()
            fabels = torch.zeros(batch_size).long().to(device)
            token_type_id = token_type_id.to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=atten, token_type_ids=token_type_id, labels=fabels)
            _, logits = outputs[:2]
            ypred = torch.argmax(logits.cpu(), dim=1)
            
            pred.extend([val.item() for val in ypred])
            actual.extend([val.item() for val in labels])
    accuracy = accuracy_score(actual, pred)
    f1 = f1_score(actual, pred, average=None)  # list of f1 scores, one for each label
    print('Accuracy:', accuracy)
    print('F1 scores:', f1)

    # Load tsv file and save results
    cols = ['model', 'accuracy', 'Yes', 'No', 'C.yes', 'Mid', 'Other', 'P.yes', 'P.no', 'Unsure', '8']
    df = None
    if os.path.exists(args.output_path):
        df = pd.read_csv(args.output_path, sep='\t')
    else: # file does not exist
        df = pd.DataFrame(columns=cols)

    row = [args.model_type, accuracy, None, None, None, None, None, None, None, None, None]
    for i in range(len(f1)):  # override None values for labels that exist
        row[i + 2] = f1[i]

    df = df.append(pd.DataFrame([row], columns=cols))
    df.to_csv(args.output_path, sep="\t", index=None)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--test_data', type=str, required=True)
    parser.add_argument('--model_type', type=str, required=True, help="choose a valid pretrained model")
    parser.add_argument('--dataset_type', type=str, required=True)
    parser.add_argument('--batch_size', default=32, type=int, help="total batch size")
    parser.add_argument('--num_labels', type=int, required=True, help="choose the number of labels for the experiment")
    parser.add_argument('--output_path', type=str, required=True, help="results output file name")
    parser.add_argument('--dataset_mode', type=str)
    args = parser.parse_args()
    
    config = BertConfig.from_pretrained(args.model_type, num_labels=args.num_labels)
    model = BertForSequenceClassification.from_pretrained(args.model_type, config=config)

    tokenizer = BertTokenizer.from_pretrained(args.model_type)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    evaluate(args, model, tokenizer, device)
