#!/bin/bash

## File for testing on the matched, strict dataset

BASE_DIR="/home/azureuser/IndirectAnswers"

cd "$BASE_DIR"


echo "Testing BOOLQ CIRCA on CIRCA with matched, relaxed labels..."

# Evaluate on test dataset
python MNLItesting.py \
--test_data "$BASE_DIR/data/CIRCA/circa-data-test.tsv" \
--model_type "$BASE_DIR/models/BOOLQ_CIRCA_BERT_matched_relaxed_e3_lr5e-5_b32" \
--dataset_type "CIRCA" \
--dataset_mode "qa" \
--batch_size "32" \
--num_labels "6" \
--output_path "$BASE_DIR/results/BOOLQ-CIRCA-test-results.tsv"

# Evaluate on dev dataset
python MNLItesting.py \
--test_data "$BASE_DIR/data/CIRCA/circa-data-dev.tsv" \
--model_type "$BASE_DIR/models/BOOLQ_CIRCA_BERT_matched_relaxed_e3_lr5e-5_b32" \
--dataset_type "CIRCA" \
--dataset_mode "qa" \
--batch_size "32" \
--num_labels "6" \
--output_path "$BASE_DIR/results/BOOLQ-CIRCA-dev-results.tsv"

echo "Finished testing BOOLQ CIRCA with matched, relaxed labels..."
