#!/bin/bash

## File for finetuning the DIS model on the unmatched, relaxed version of the CIRCA dataset

BASE_DIR="/home/azureuser/IndirectAnswers"  # IndirectAnswers directory
UNMATCHED_DIR="$BASE_DIR/data/CIRCA/unmatched/scenario"
OUTPUT_DIR="$BASE_DIR/models/"
MODEL_TYPE="/home/azureuser/IndirectAnswers/models/DIS_BERT_e3_lr2e-5_b16"

cd "$BASE_DIR"

ScenarioNum="0 1 2 3 4 5 6 7 8 9"

## Modify the script so that we can make a new directory for the different BERT experiments
## For label of size 6
echo "Finetuning DIS CIRCA with unmatched, relaxed labels..."

# Running through the different modes, using their best hyperparameters
lr="5e-5"
for j in $ScenarioNum; do # 10 iterations, leaving a different scenario out each time
    echo "STARTING... Learning rate: $lr, Scenario: $j"
    python finetuning.py \
        --train_data "${UNMATCHED_DIR}${j}/circa-data-train.tsv" \
        --dev_data "$UNMATCHED_DIR${j}/circa-data-dev.tsv" \
        --model_name "DIS_CIRCA_BERT_unmatched_relaxed_s${j}_e3_lr${lr}_b32" \
        --dataset_type "CIRCA" \
        --transfer_on \
        --batch_size "32" \
        --epochs "3" \
        --learning_rate "$lr" \
        --output_dir "$OUTPUT_DIR" \
        --model_type "$MODEL_TYPE" \
        --num_labels "6" \
        --prev_labels "5"
done

echo "Finished finetuning DIS CIRCA with unmatched, relaxed labels..."
