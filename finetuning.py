from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel, AdamW, BertTokenizer, BertForSequenceClassification, BertConfig

import os
import torch
import logging
import torch
import shutil
import pandas as pd
import numpy as np
import dataloader

# Import the dataset to use here

log_format = '%(asctime)-10s: %(message)s'
logging.basicConfig(level=logging.INFO, format=log_format)

# Runs validation to find the best model
def validate(args, model, tokenizer, device, epoch, min_loss, model_path):
    logging.info("***** Running development *****")
    
    # Loads a dataset depending on the number
    # 1: Circa, 2: BoolQ, 3: MNLI, 4: DIS
    dev_dataloader = None
    if args.dataset_type == 'CIRCA':
        dev_dataloader = dataloader.getCircaDataloader(args.dev_data, batch_size=args.batch_size, num_workers=4, tokenizer=tokenizer, mode=args.dataset_mode)
    elif args.dataset_type == 'BOOLQ':
        dev_dataloader = dataloader.getBOOLQDataloader(args.dev_data, batch_size=args.batch_size, num_workers=4, tokenizer=tokenizer)
    elif args.dataset_type == 'MNLI':
        dev_dataloader = dataloader.getMNLIDataloader(args.dev_data, batch_size=args.batch_size, num_workers=4, tokenizer=tokenizer)
    else:
        dev_dataloader = dataloader.getDISDataloader(args.dev_data, batch_size=args.batch_size, num_workers=4, tokenizer=tokenizer)
    
    dev_loss = 0.0
    nb_dev_step = 0

    model.eval()

    for batch in tqdm(dev_dataloader, desc="Checking dev model accuracy..."):
        with torch.no_grad():
            if args.dataset_type == 'CIRCA':
                # Strict case
                if args.num_labels == 9:
                    input_ids, atten, labels, token_type_id = batch['input_ids'], batch['attention_mask'], batch['goldstandard1'], batch['token_type_ids']
                # Relaxed case
                else:
                    input_ids, atten, labels, token_type_id = batch['input_ids'], batch['attention_mask'], batch['goldstandard2'], batch['token_type_ids']
            elif args.dataset_type == 'BOOLQ':
                input_ids, atten, labels, token_type_id = batch['input_ids'], batch['attention_mask'], batch['answer'], batch['token_type_ids']
            elif args.dataset_type == 'MNLI':
                input_ids, atten, labels, token_type_id = batch['sentence_input_ids'], batch['sentence_attention_mask'], batch['gold_labels'], batch['sentence_token_type_ids']
            else:
                input_ids, atten, labels, token_type_id = batch['input_ids'], batch['attention_mask'], batch['labels'], batch['token_type_ids']

            input_ids = input_ids.to(device)
            atten = atten.to(device)
            labels = labels.to(device).squeeze()
            token_type_id = token_type_id.to(device)

            outputs = model(input_ids=input_ids, attention_mask=atten, token_type_ids=token_type_id, labels=labels)

            tmp_dev_loss, _ = outputs[:2]
            if args.multi_gpu:
                tmp_dev_loss = torch.mean(tmp_dev_loss)

            dev_loss += tmp_dev_loss.item()
            nb_dev_step += 1

    loss = dev_loss / nb_dev_step
    print("Validation loss:", loss)
    
    if loss < min_loss:
        # Saving a trained model
        logging.info("** ** * Saving validated model ** ** * ")
        model_to_save = model.module if hasattr(model, 'module') else model

        output_model_file = os.path.join(model_path, "pytorch_model.bin")
        output_config_file = os.path.join(model_path, "config.json")

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(model_path)

        # New minimum loss value
        return loss

    # Old minimum loss value
    return min_loss

def main():
    parser = ArgumentParser()
    parser.add_argument('--train_data', type=Path, required=True)
    parser.add_argument('--dev_data', type=Path, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--dataset_type', type=str, required=True)
    parser.add_argument('--multi_gpu_on', dest='multi_gpu', action='store_true')
    parser.add_argument('--multi_gpu_off', dest='multi_gpu', action='store_false')
    parser.add_argument('--transfer_on', dest='transfer_learning', action='store_true')
    parser.add_argument('--transfer_off', dest='transfer_learning', action='store_false')
    parser.add_argument('--epochs', type=int, default=20, help="number of epochs to train for")
    parser.add_argument('--model_type', type=str, required=True, help="choose a valid pretrained model")
    parser.add_argument('--batch_size', default=32, type=int, help="total batch size")
    parser.add_argument('--learning_rate', default=1e-5, type=float, help="initial learning rate for Adam")
    parser.add_argument('--grad_clip', type=float, default=0.25, help="Grad clipping value")
    parser.add_argument('--num_labels', type=int, required=True, help="choose the number of labels for the experiment")
    parser.add_argument("--prev_labels", type=int, default=0, help="previous number of labels")
    parser.add_argument('--dataset_mode', type=str)
    
    parser.set_defaults(multi_gpu=False)
    parser.set_defaults(transfer_learning=False)

    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model_path = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)    

    if args.prev_labels != 0:
        config = BertConfig.from_pretrained(args.model_type, num_labels=args.prev_labels) 
    else:
        config = BertConfig.from_pretrained(args.model_type, num_labels=args.num_labels)
    

    tokenizer = BertTokenizer.from_pretrained(args.model_type)        
    
    model = BertForSequenceClassification.from_pretrained(args.model_type, config=config)
    if args.transfer_learning:
        model.classifier = torch.nn.Linear(768, args.num_labels)
        model.num_labels = args.num_labels

    if args.multi_gpu:
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #device = "cpu"

    model.zero_grad()
    model.to(device)

    params = [p for n,p in model.named_parameters()]
    optimizer = AdamW(params, lr=args.learning_rate)

    logging.info("****** Running training *****")
    logging.info(f"  Num epochs = {args.epochs}")
    logging.info(f"  Learning rate = {args.learning_rate}")
    total_loss = 0
    global_step = 0
    min_loss = float('inf')

    train_iterator = trange(0, args.epochs, desc="Epoch")
    for epoch, _ in enumerate(train_iterator):
        # 1: Circa, 2: BoolQ, 3: MNLI, 4: DIS
        train_dataloader = None
        if args.dataset_type == 'CIRCA':
            train_dataloader = dataloader.getCircaDataloader(args.train_data, batch_size=args.batch_size, num_workers=1, tokenizer=tokenizer, mode=args.dataset_mode)
        elif args.dataset_type == 'BOOLQ':
            print("BOOLQ")
            train_dataloader = dataloader.getBOOLQDataloader(args.train_data, batch_size=args.batch_size, num_workers=1, tokenizer=tokenizer)
        elif args.dataset_type == 'MNLI':
            train_dataloader = dataloader.getMNLIDataloader(args.train_data, batch_size=args.batch_size, num_workers=1, tokenizer=tokenizer)
        else:
            train_dataloader = dataloader.getDISDataloader(args.train_data, batch_size=args.batch_size, num_workers=1, tokenizer=tokenizer)

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")                                       
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()
        
        for step, batch in enumerate(epoch_iterator):
            if args.dataset_type == 'CIRCA':
                # Strict case
                if args.num_labels == 9:
                    input_ids, atten, labels, token_type_id = batch['input_ids'], batch['attention_mask'], batch['goldstandard1'], batch['token_type_ids']
                # Relaxed case
                else:
                    input_ids, atten, labels, token_type_id = batch['input_ids'], batch['attention_mask'], batch['goldstandard2'], batch['token_type_ids']
            elif args.dataset_type == 'BOOLQ':
                input_ids, atten, labels, token_type_id = batch['input_ids'], batch['attention_mask'], batch['answer'], batch['token_type_ids']
            elif args.dataset_type == 'MNLI':
                input_ids, atten, labels, token_type_id = batch['sentence_input_ids'], batch['sentence_attention_mask'], batch['gold_labels'], batch['sentence_token_type_ids']
            else:
                input_ids, atten, labels, token_type_id = batch['input_ids'], batch['attention_mask'], batch['labels'], batch['token_type_ids']

            input_ids = input_ids.to(device)
            atten = atten.to(device)
            labels = labels.to(device).squeeze()
            token_type_id = token_type_id.to(device)

            outputs = model(input_ids=input_ids, token_type_ids=token_type_id, attention_mask=atten, labels=labels)

            loss = outputs[0]                                                                                                                                               
            # Average on multi-gpu training
            if args.multi_gpu:
                loss = torch.mean(loss)

            loss.backward()

            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            model.zero_grad()

            global_step += 1
            epoch_iterator.set_description(f"Loss: {total_loss / (global_step + 1)}")
	            
        min_loss = validate(args, model, tokenizer, device, epoch, min_loss, model_path)

if __name__ == '__main__':
    main()
