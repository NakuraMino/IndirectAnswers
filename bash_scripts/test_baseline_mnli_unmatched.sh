#!/bin/bash

## File for testing on the unmatched, relaxed dataset

BASE_DIR="/home/azureuser/IndirectAnswers"  # IndirectAnswers directory
UNMATCHED_DIR="$BASE_DIR/data/CIRCA/unmatched/scenario"

cd "$BASE_DIR"

ScenarioNum="0 1 2 3 4 5 6 7 8 9"

echo "Testing MNLI only on CIRCA with unmatched, strict labels..."

for j in $ScenarioNum; do # 10 iterations, leaving a different scenario out each time
    lr="2e-5"
    echo "STARTING... Scenario: $j"
    echo "evaluating on test dataset"
    # Evaluate on test dataset
    python MNLItesting.py \
    --test_data "${UNMATCHED_DIR}${j}/circa-data-test.tsv" \
    --model_type "$BASE_DIR/models/MNLI_BERT_e3_lr2e-5_b16" \
    --dataset_type "CIRCA" \
    --batch_size "32" \
    --num_labels "4" \
    --output_path "$BASE_DIR/results/MNLI-test-results.tsv"
    
    echo "evaluating on dev dataset"
    # Evaluate on dev dataset
    python MNLItesting.py \
    --test_data "${UNMATCHED_DIR}${j}/circa-data-dev.tsv" \
    --model_type "$BASE_DIR/models/MNLI_BERT_e3_lr2e-5_b16" \
    --dataset_type "CIRCA" \
    --batch_size "32" \
    --num_labels "4" \
    --output_path "$BASE_DIR/results/MNLI-dev-results.tsv"
done

echo "Finished testing CIRCA with unmatched, relaxed labels..."
