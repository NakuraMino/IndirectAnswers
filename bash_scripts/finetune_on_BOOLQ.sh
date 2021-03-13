#!/bin/bash

## File for finetuning on the relaxed dataset

#BASE_DIR="/data2/limill01/IndirectAnswers/"
#OUTPUT_DIR="/data2/limill01/IndirectAnswers/models/"
BASE_DIR="home/azureuser/IndirectAnswers/"
OUTPUT_DIR="/home/azureuser/IndirectAnswers/models"
MODEL_TYPE="bert-base-cased"

cd "$BASE_DIR"

LearningRates="5e-5 3e-5 2e-5"
Epochs="2 3 4"
BatchSize="16 32"

## Modify the script so that we can make a new directory for the different BERT experiments
## For label of size 6
echo "Finetuning on BOOLQ..."

python finetuning.py \
--train_data "$BASE_DIR/data/BoolQ/train.jsonl" \
--dev_data "$BASE_DIR/data/BoolQ/dev.jsonl" \
--model_name "BOOLQ_BERT_e4_lr3e-5_b16" \
--dataset_type "BOOLQ" \
--multi_gpu_on \
--batch_size "16" \
--epochs "4" \
--learning_rate " 3e-5" \
--output_dir "$OUTPUT_DIR" \
--model_type "$MODEL_TYPE" \
--num_labels "2"

echo "Finished finetuning on BOOLQ..."
