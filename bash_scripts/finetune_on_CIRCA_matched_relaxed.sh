#!/bin/bash

## File for finetuning on the matched, relaxed dataset

BASE_DIR=".."  # IndirectAnswers directory
OUTPUT_DIR="$BASE_DIR/models/"
MODEL_TYPE="bert-base-cased"

cd "$BASE_DIR"

Modes=('q' 'a' 'qa') # q=question, a=answer
LearningRates=('3e-5' '2e-5' '3e-5')

## Modify the script so that we can make a new directory for the different BERT experiments
## For label of size 6
echo "Finetuning on CIRCA with matched, relaxed labels..."

# Running through the different modes, using their best hyperparameters
for i in ${!Modes[@]}; do
    mode=${Modes[i]}
    lr=${LearningRates[i]}
    echo "STARTING... Mode: $mode, Learning rate: $lr"
    python finetuning.py \
        --train_data "$BASE_DIR/data/circa-data-train.tsv" \
        --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
        --model_name "CIRCA_BERT_matched_relaxed_e3_lr${lr}_b32" \
        --dataset_type "CIRCA" \
        --batch_size "32" \
        --epochs "3" \
        --learning_rate "$lr" \
        --output_dir "$OUTPUT_DIR" \
        --model_type "$MODEL_TYPE" \
        --num_labels "6"
done

echo "Finished finetuning CIRCA with matched, relaxed labels..."
