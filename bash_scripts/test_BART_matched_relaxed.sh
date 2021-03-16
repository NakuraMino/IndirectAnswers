#!/bin/bash

## File for testing on the matched, relaxed dataset

BASE_DIR="/home/azureuser/IndirectAnswers"

cd "$BASE_DIR"

echo "Testing BART on CIRCA with matched, relaxed labels..."

# Evaluate on test dataset
python bartTesting.py \
    --test_data "$BASE_DIR/data/CIRCA/circa-data-test.tsv" \
    --model_type "$BASE_DIR/models/BART_matched_strict_e3_lr3e-5_b32" \
    --dataset_type "CIRCA" \
    --batch_size "32" \
    --num_labels "6" \
    --output_path "$BASE_DIR/results/BART-test-results.tsv"

    # Evaluate on dev dataset
python bartTesting.py \
    --test_data "$BASE_DIR/data/CIRCA/circa-data-dev.tsv" \
    --model_type "$BASE_DIR/models/BART_matched_strict_e3_lr3e-5_b32" \
    --dataset_type "CIRCA" \
    --batch_size "32" \
    --num_labels "6" \
    --output_path "$BASE_DIR/results/BART-dev-results.tsv"

echo "Finished testing BART CIRCA with matched, strict labels..."
