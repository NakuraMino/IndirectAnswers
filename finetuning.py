from argparse import ArgumentParser
from pathlib import Path
from tqdm import tqdm, trange
from transformers import AutoTokenizer, AutoModel, AdamW, BertTokenizer, BertForSequenceClassification, BertConfig

import os
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

    dev_dataloader = dataloader.getCircaDataloader(args.dev_data, batch_size=args.batch_size, num_workers=4, use_tokenizer=False, tokenizer=tokenizer)

    dev_loss = 0.0
    nb_dev_step = 0

    model.eval()

    for batch in tqdm(dev_dataloader, desc="Checking dev model accuracy..."):
        with torch.no_grad():
            input_ids, atten, labels, token_type_id = batch['input_ids'], batch['attention_mask'], batch['goldstandard1'], batch['token_type_ids']

            input_ids = input_ids.to(device)
            atten = atten.to(device)
            labels = labels.to(device).squeeze()
            token_type_id = token_type_id.to(device)

            outputs = model(input_ids=input_ids, attention_mask=atten, token_type_ids=token_type_id, labels=labels)
            tmp_dev_loss, _ = outputs[:2]

            dev_loss += tmp_dev_loss.item()
            nb_dev_step += 1

    loss = dev_loss / nb_dev_step
    print("Validation loss:", loss)
    
    if loss < min_loss:
        min_loss = loss

        # Saving a trained model
        logging.info("** ** * Saving validated model ** ** * ")
        model_to_save = model.module if hasattr(model, 'module') else model

        output_model_file = os.path.join(model_path, "pytorch_model.bin")
        output_config_file = os.path.join(model_path, "config.json")

        torch.save(model_to_save.state_dict(), output_model_file)
        model_to_save.config.to_json_file(output_config_file)
        tokenizer.save_vocabulary(model_path)

def main():
    parser = ArgumentParser()
    parser.add_argument('--train_data', type=Path, required=True)
    parser.add_argument('--dev_data', type=Path, required=True)
    parser.add_argument('--test_data', type=Path, required=True)
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    parser.add_argument('--type', type=bool, required=True, help="relaxed vs. strict case")
    parser.add_argument('--epochs', type=int, default=20, help="number of epochs to train for")
    parser.add_argument('--model_type', type=str, required=True, help="choose a valid pretrained model")
    parser.add_argument('--batch_size', default=32, type=int, help="total batch size")
    parser.add_argument('--learning_rate', default=1e-5, type=float, help="initial learning rate for Adam")
    parser.add_argument('--grad_clip', type=float, default=0.25, help="Grad clipping value")
    parser.add_argument('--num_labels', type=int, required=True, help="choose the number of labels for the experiment")

    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model_path = os.path.join(args.output_dir, args.model_name)
    if not os.path.exists(model_path):
        os.mkdir(model_path)    

    config = BertConfig.from_pretrained(args.model_type, num_labels=args.num_labels)

    tokenizer = BertTokenizer.from_pretrained(args.model_type)
    model = BertForSequenceClassification.from_pretrained(args.model_type, config=config)
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

    train_iterator = trange(0, args.epochs, desc="Epoch")
    for epoch, _ in enumerate(train_iterator):
        # epoch_dataset = dataset_train                                                                   
        # train_dataloader = DataLoader(
        #   	epoch_dataset, 
        #     batch_size=args.train_batch_size, 
        #     shuffle=False,
        # )
        #train_dataloader = dataloader.getCircaDataloader('./data/circa-data.tsv', batch_size=1, num_workers=1, use_tokenizer=False)
        train_dataloader = dataloader.getCircaDataloader(args.train_data, batch_size=args.batch_size, num_workers=1, use_tokenizer=False, tokenizer=tokenizer)

        epoch_iterator = tqdm(train_dataloader, desc="Iteration")                                       
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()
       
         
        for step, batch in enumerate(epoch_iterator):
            # Strict case
            if args.type:
                input_ids, atten, labels, token_type_id = batch['input_ids'], batch['attention_mask'], batch['goldstandard1'], batch['token_type_ids']
            # Relaxed case
            else:
                input_ids, atten, labels, token_type_id = batch['input-ids'], batch['attention_mask'], batch['goldstandard2'], batch['token_type_ids']

            input_ids = input_ids.to(device)
            atten = atten.to(device)
            labels = labels.to(device).squeeze()
            token_type_id = token_type_id.to(device)

            #input_ids = torch.reshape(input_ids, (1, -1))
            #atten = torch.reshape(atten, (1, -1))
            #token_type_id = torch.reshape(token_type_id, (1, -1))

            outputs = model(input_ids=input_ids, token_type_ids=token_type_id, attention_mask=atten, labels=labels)

            loss = outputs[0]                                                                                                                                               
            loss.backward()

            total_loss += loss.item()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()
            model.zero_grad()

            global_step += 1
            epoch_iterator.set_description(f"Loss: {total_loss / (global_step + 1)}")
         
        min_loss = float('inf')
        validate(args, model, tokenizer, device, epoch, min_loss, model_path)

if __name__ == '__main__':
    main()
