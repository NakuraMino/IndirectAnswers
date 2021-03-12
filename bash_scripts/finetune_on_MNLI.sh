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

python finetuning.py \
--train_data "$BASE_DIR/data/multinli_1.0/multinli_1.0_train.jsonl" \
--dev_data "$BASE_DIR/data/multinli_1.0/multinli_1.0_dev_matched.jsonl" \
--model_name "MNLI_BERT_relaxed_e${e}_lr${lr}_b${b}" \
--dataset_type "3" \
--batch_size "16" \
--epochs "3" \
--learning_rate " 2e-5" \
--output_dir "$OUTPUT_DIR" \
--model_type "$MODEL_TYPE" \
--num_labels "3"

# Running through the different combinations
#for lr in $LearningRates; do
#    for e in $Epochs; do
#        for b in $BatchSize; do
#            echo "lr: $lr, epoch: $e, batch: $b"
#            python finetuning.py \
#                --train_data "$BASE_DIR/data/multinli_1.0/multinli_1.0_train.jsonl" \
#                --dev_data "$BASE_DIR/data/multinli_1.0/multinli_1.0_dev_matched.jsonl" \
#                --model_name "MNLI_BERT_relaxed_e${e}_lr${lr}_b${b}" \
#                --dataset_type "3" \
#                --batch_size "$b" \
#                --epochs "$e" \
#                --learning_rate "$lr" \
#                --output_dir "$OUTPUT_DIR" \
#                --model_type "$MODEL_TYPE" \
#                --num_labels "3"
#        done
#    done
#done

echo "Finished finetuning on MNLI..."
