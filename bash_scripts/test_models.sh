#!/bin/bash

## File for testing on the datasets

BASE_DIR="/data2/limill01/IndirectAnswers"
OUTPUT_DIR="/data2/limill01/IndirectAnswers/results"
MODEL_TYPE="bert-base-cased"

cd "$BASE_DIR"

## Modify the script so that we can make a new directory for the different BERT experiments
## For label of size 9
echo "Testing on CIRCA..."
python testing.py \
    --model_path "$BASE_DIR/models" \  # TODO: put real path
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \    
