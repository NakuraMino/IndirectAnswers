#!/bin/bash

## File for testing on the matched, strict dataset

BASE_DIR="/home/azureuser/IndirectAnswers"

cd "$BASE_DIR"


echo "Testing DIS CIRCA on CIRCA with matched, strict labels..."

# Evaluate on test dataset
python MNLItesting.py \
--test_data "$BASE_DIR/data/CIRCA/circa-data-test.tsv" \
--model_type "$BASE_DIR/models/DIS_CIRCA_BERT_matched_strict_e3_lr3e-5_b32" \
--dataset_type "CIRCA" \
--dataset_mode "qa" \
--batch_size "32" \
--num_labels "9" \
--output_path "$BASE_DIR/results/DIS-CIRCA-test-results.tsv"

# Evaluate on dev dataset
python MNLItesting.py \
--test_data "$BASE_DIR/data/CIRCA/circa-data-dev.tsv" \
--model_type "$BASE_DIR/models/DIS_CIRCA_BERT_matched_strict_e3_lr3e-5_b32" \
--dataset_type "CIRCA" \
--dataset_mode "qa" \
--batch_size "32" \
--num_labels "9" \
--output_path "$BASE_DIR/results/DIS-CIRCA-dev-results.tsv"

echo "Finished testing DIS CIRCA with matched, strict labels..."
