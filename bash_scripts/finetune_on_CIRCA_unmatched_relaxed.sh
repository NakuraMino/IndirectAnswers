#!/bin/bash

## File for finetuning on the unmatched, relaxed dataset

BASE_DIR=".."  # IndirectAnswers directory
OUTPUT_DIR="$BASE_DIR/results/"
MODEL_TYPE="bert-base-cased"

cd "$BASE_DIR"

Modes=('q' 'a' 'qa') # q=question, a=answer
LearningRates=('3e-5' '2e-5' '3e-5')
ScenarioNum="0 1 2 3 4 5 6 7 8 9"

## Modify the script so that we can make a new directory for the different BERT experiments
## For label of size 6
echo "Finetuning on CIRCA with unmatched, relaxed labels..."

# Running through the different modes, using their best hyperparameters
for i in ${!Modes[@]}; do
    for j in $ScenarioNum; do # 10 iterations, leaving a different scenario out each time
        mode=${Modes[i]}
        lr=${LearningRates[i]}
        echo "STARTING... Mode: $mode, Learning rate: $lr, Iter: $j"
        python finetuning.py \
            --train_data "$BASE_DIR/data/unmatched-circa-data-train.tsv" \
            --dev_data "$BASE_DIR/data/unmatched-circa-data-dev.tsv" \
            --model_name "CIRCA_BERT_unmatched_relaxed_e3_lr${lr}_b32" \
            --dataset_type "CIRCA" \
            --batch_size "32" \
            --epochs "3" \
            --learning_rate "$lr" \
            --output_dir "$OUTPUT_DIR" \
            --model_type "$MODEL_TYPE" \
            --num_labels "6"
    done
done

echo "Finished finetuning CIRCA with unmatched, relaxed labels..."
