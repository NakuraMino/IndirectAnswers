#!/bin/bash

## File for testing on the matched, strict dataset

BASE_DIR="/home/azureuser/IndirectAnswers"

cd "$BASE_DIR"


echo "Testing MNLI on CIRCA with matched, strict labels..."

# Evaluate on test dataset
python MNLItesting.py \
--test_data "$BASE_DIR/data/CIRCA/circa-data-test.tsv" \
--model_type "$BASE_DIR/models/MNLI_BERT_e3_lr2e-5_b16" \
--dataset_type "CIRCA" \
--dataset_mode "qa" \
--batch_size "32" \
--num_labels "2" \
--output_path "$BASE_DIR/results/CIRCA-test-results.tsv"

# Evaluate on dev dataset
python MNLItesting.py \
--test_data "$BASE_DIR/data/CIRCA/circa-data-dev.tsv" \
--model_type "$BASE_DIR/models/MNLI_BERT_e3_lr2e-5_b16" \
--dataset_type "CIRCA" \
--dataset_mode "qa" \
--batch_size "32" \
--num_labels "2" \
--output_path "$BASE_DIR/results/CIRCA-dev-results.tsv"

echo "Finished testing CIRCA with matched, strict labels..."
