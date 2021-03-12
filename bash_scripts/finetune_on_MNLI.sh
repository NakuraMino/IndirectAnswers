#!/bin/bash

## File for finetuning on the relaxed dataset

BASE_DIR="/data2/limill01/IndirectAnswers/"
OUTPUT_DIR="/data2/limill01/IndirectAnswers/results/"
MODEL_TYPE="bert-base-cased"

cd "$BASE_DIR"

LearningRates="5e-5 3e-5 2e-5"
Epochs="2 3 4"
BatchSize="16 32"

## Modify the script so that we can make a new directory for the different BERT experiments
## For label of size 6
echo "Finetuning on MNLI..."

# Running through the different combinations
for lr in $LearningRates; do
    for e in $Epochs; do
        for b in $BatchSize; do
            echo "lr: $lr, epoch: $e, batch: $b"
            python finetuning.py \
                --train_data "$BASE_DIR/data/circa-data-train.tsv" \
                --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
                --model_name "CIRCA_BERT_relaxed_e${e}_lr${lr}_b${b}" \
                --dataset_type "3" \
                --batch_size "$b" \
                --epochs "$e" \
                --learning_rate "$lr" \
                --output_dir "$OUTPUT_DIR" \
                --model_type "$MODEL_TYPE" \
                --num_labels "6"
        done
    done
done

echo "Finished finetuning on MNLI..."
