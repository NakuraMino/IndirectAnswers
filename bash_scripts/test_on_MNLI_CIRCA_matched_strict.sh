#!/bin/bash

## File for testing on the matched, relaxed dataset

BASE_DIR="/home/azureuser/IndirectAnswers"

cd "$BASE_DIR"

echo "Testing on CIRCA with matched, relaxed labels..."

# Evaluate on test dataset
python testing.py \
    --test_data "$BASE_DIR/data/CIRCA/circa-data-test.tsv" \
    --model_type "$BASE_DIR/models/MNLI_CIRCA_BERT_matched_strict_e3_lr2e-5_b32" \
    --dataset_type "CIRCA" \
    --batch_size "32" \
    --num_labels "9" \
    --output_path "$BASE_DIR/results/MNLI-test-results.tsv"

    # Evaluate on dev dataset
python testing.py \
    --test_data "$BASE_DIR/data/CIRCA/circa-data-dev.tsv" \
    --model_type "$BASE_DIR/models/MNLI_CIRCA_BERT_matched_strict_e3_lr2e-5_b32" \
    --dataset_type "CIRCA" \
    --batch_size "32" \
    --num_labels "9" \
    --output_path "$BASE_DIR/results/MNLI-dev-results.tsv"

echo "Finished testing CIRCA with matched, relaxed labels..."
