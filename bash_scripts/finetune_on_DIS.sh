#!/bin/bash

## File for finetuning on the DIS dataset

#BASE_DIR="/data2/limill01/IndirectAnswers/"
#OUTPUT_DIR="/data2/limill01/IndirectAnswers/results/
BASE_DIR="/home/azureuser/IndirectAnswers/"
OUTPUT_DIR="/home/azureuser/IndirectAnswers/models"
MODEL_TYPE="bert-base-cased"

cd "$BASE_DIR"

LearningRates="5e-5 3e-5 2e-5"
Epochs="2 3 4"
BatchSize="16 32"

## Modify the script so that we can make a new directory for the different BERT experiments
## For label of size 6

echo "Finetuning on DIS..."

python finetuning.py \
--train_data "$BASE_DIR/data/DIS/discourse_EN_FIVE_and_but_because_if_when_2017dec12_train.tsv" \
--dev_data "$BASE_DIR/data/DIS/discourse_EN_FIVE_and_but_because_if_when_2017dec12_valid.tsv" \
--model_name "DIS_BERT_e3_lr2e-5_b16" \
--dataset_type "DIS" \
--multi_gpu_on \
--batch_size "32" \
--epochs "2" \
--learning_rate " 2e-5" \
--output_dir "$OUTPUT_DIR" \
--model_type "$MODEL_TYPE" \
--num_labels "5"

echo "Finished finetuning on DIS..."
