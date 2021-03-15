#!/bin/bash

## File for testing on the matched, relaxed dataset

BASE_DIR="/home/azureuser/IndirectAnswers"

cd "$BASE_DIR"

Modes=('q' 'a' 'qa') # q=question, a=answer
LearningRates=('3e-5' '2e-5' '3e-5')

echo "Testing on CIRCA with matched, relaxed labels..."

for i in ${!Modes[@]}; do
    mode=${Modes[i]}
    lr=${LearningRates[i]}
    echo "STARTING... Mode: $mode, Learning rate: $lr"
    # Evaluate on test dataset
    python testing.py \
        --test_data "$BASE_DIR/data/CIRCA/circa-data-test.tsv" \
        --model_type "$BASE_DIR/models/CIRCA_BERT_matched_relaxed_${mode}_e3_lr${lr}_b32" \
        --dataset_type "CIRCA" \
        --dataset_mode "$mode" \
        --batch_size "32" \
        --num_labels "6" \
        --output_path "$BASE_DIR/results/CIRCA-test-results.tsv"

    # Evaluate on dev dataset
    python testing.py \
        --test_data "$BASE_DIR/data/CIRCA/circa-data-dev.tsv" \
        --model_type "$BASE_DIR/models/CIRCA_BERT_matched_relaxed_${mode}_e3_lr${lr}_b32" \
        --dataset_type "CIRCA" \
        --dataset_mode "$mode" \
        --batch_size "32" \
        --num_labels "6" \
        --output_path "$BASE_DIR/results/CIRCA-dev-results.tsv"
done

echo "Finished testing CIRCA with matched, relaxed labels..."
