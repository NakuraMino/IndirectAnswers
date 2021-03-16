#!/bin/bash

## File for finetuning on the matched, strict dataset

#BASE_DIR=".."  # IndirectAnswers directory
BASE_DIR="/home/azureuser/IndirectAnswers"
OUTPUT_DIR="$BASE_DIR/models/"
MODEL_TYPE="facebook/bart-base"

cd "$BASE_DIR"


## Modify the script so that we can make a new directory for the different BERT experiments
## For label of size 9
echo "Finetuning on CIRCA with matched, relaxed labels..."

# Running through the different modes, using their best hyperparameters

lr="3e-5"

echo "STARTING... Learning rate: $lr"
python bartFinetuning.py \
    --train_data "$BASE_DIR/data/CIRCA/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/CIRCA/circa-data-dev.tsv" \
    --model_name "BART_matched_strict_e3_lr${lr}_b32" \
    --dataset_type "CIRCA" \
    --batch_size "32" \
    --epochs "3" \
    --learning_rate "$lr" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "6"

echo "Finished finetuning CIRCA with matched, relaxed labels..."
