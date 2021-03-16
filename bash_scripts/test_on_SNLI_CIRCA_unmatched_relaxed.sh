#!/bin/bash

## File for testing on the unmatched, relaxed dataset

BASE_DIR="/home/azureuser/IndirectAnswers"  # IndirectAnswers directory
UNMATCHED_DIR="$BASE_DIR/data/CIRCA/unmatched/scenario"

cd "$BASE_DIR"

ScenarioNum="0 1 2 3 4 5 6 7 8 9"

echo "Testing on SNLI (pretrained) CIRCA with unmatched, relaxed labels..."

lr="5e-5"
for j in $ScenarioNum; do # 10 iterations, leaving a different scenario out each time
    echo "STARTING... Mode: $mode, Learning rate: $lr, Scenario: $j"
    echo "evaluating on test dataset"
    # Evaluate on test dataset
    python testing.py \
    --test_data "${UNMATCHED_DIR}${j}/circa-data-test.tsv" \
    --model_type "$BASE_DIR/models/SNLI_CIRCA_BERT_unmatched_relaxed_s${j}_e3_lr${lr}_b32" \
    --dataset_type "CIRCA" \
    --batch_size "32" \
    --num_labels "6" \
    --output_path "$BASE_DIR/results/SNLI-test-results.tsv"
    
    echo "evaluating on dev dataset"
    # Evaluate on dev dataset
    python testing.py \
    --test_data "${UNMATCHED_DIR}${j}/circa-data-dev.tsv" \
    --model_type "$BASE_DIR/models/SNLI_CIRCA_BERT_unmatched_relaxed_s${j}_e3_lr${lr}_b32" \
    --dataset_type "CIRCA" \
    --batch_size "32" \
    --num_labels "6" \
    --output_path "$BASE_DIR/results/SNLI-dev-results.tsv"
done

echo "Finished testing SNLI (pretrained) CIRCA with unmatched, relaxed labels..."
