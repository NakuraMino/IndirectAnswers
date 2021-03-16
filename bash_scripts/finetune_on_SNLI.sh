#!/bin/bash

## File for finetuning on the SNLI dataset

BASE_DIR="/home/azureuser/IndirectAnswers/"
OUTPUT_DIR="/home/azureuser/IndirectAnswers/models"
MODEL_TYPE="bert-base-cased"

cd "$BASE_DIR"

echo "Finetuning on SNLI..."

python finetuning.py \
--train_data "$BASE_DIR/data/snli_1.0/snli_1.0_train.jsonl" \
--dev_data "$BASE_DIR/data/snli_1.0/snli_1.0_dev.jsonl" \
--model_name "SNLI_BERT_e3_lr2e-5_b16" \
--dataset_type "MNLI" \
--multi_gpu_on \
--batch_size "16" \
--epochs "3" \
--learning_rate " 2e-5" \
--output_dir "$OUTPUT_DIR" \
--model_type "$MODEL_TYPE" \
--num_labels "4"

echo "Finished finetuning on SNLI..."
