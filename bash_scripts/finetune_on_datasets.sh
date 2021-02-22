#!/bin/bash

## File for finetuning on the datasets

BASE_DIR="/data2/limill01/IndirectAnswers"
OUTPUT_DIR="/data2/limill01/IndirectAnswers/results"
MODEL_TYPE="bert-base-cased"

cd "$BASE_DIR"

## For label of size 9
echo "Finetuning on CIRCA..."
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_type "$MODEL_TYPE" \
    --num_labels "9"
    
