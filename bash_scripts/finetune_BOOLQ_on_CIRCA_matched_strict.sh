#!/bin/bash

## File for finetuning the BOOLQ model on the matched, strict version of the CIRCA dataset

#BASE_DIR=".."  # IndirectAnswers directory
#BASE_DIR="/data2/limill01/IndirectAnswers"
BASE_DIR="/home/azureuser/IndirectAnswers"
OUTPUT_DIR="$BASE_DIR/models/"
MODEL_TYPE="/home/azureuser/IndirectAnswers/models/BOOLQ_BERT_e4_lr3e-5_b16"

cd "$BASE_DIR"


## Modify the script so that we can make a new directory for the different BERT experiments
## For label of size 9
echo "Finetuning on CIRCA with matched, strict labels..."

# Running through the different modes, using their best hyperparameters
lr="5e-5"
echo "STARTING... Learning rate: $lr"
python finetuning.py \
    --train_data "$BASE_DIR/data/CIRCA/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/CIRCA/circa-data-dev.tsv" \
    --model_name "BOOLQ_CIRCA_BERT_matched_strict_e3_lr${lr}_b32" \
    --dataset_type "CIRCA" \
    --multi_gpu_on \
    --transfer_on \
    --batch_size "32" \
    --epochs "3" \
    --learning_rate "$lr" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "9" \
    --prev_labels "2"

echo "Finished finetuning BOOLQ CIRCA with matched, strict labels..."
