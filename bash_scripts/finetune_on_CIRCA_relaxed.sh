#!/bin/bash

## File for finetuning on the relaxed dataset

BASE_DIR="/data2/limill01/IndirectAnswers/"
OUTPUT_DIR="/data2/limill01/IndirectAnswers/results/"
MODEL_TYPE="bert-base-cased"

cd "$BASE_DIR"

## Modify the script so that we can make a new directory for the different BERT experiments
## For label of size 6
echo "Finetuning on CIRCA with relaxed labels..."

# Epochs: 2, Learning rate: 5e-5, train batch size: 16
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_name "CIRCA_BERT_relaxed_e2_lr5e-5_b16" \
    --type false \
    --batch_size "16" \
    --epochs "2" \
    --learning_rate " 5e-5" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "6"

# Epochs: 3, Learning rate: 5e-5, train batch size: 16
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_name "CIRCA_BERT_relaxed_e3_lr5e-5_b16" \
    --type false\
    --batch_size "16" \
    --epochs "3" \
    --learning_rate " 5e-5" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "6"

# Epochs: 4, Learning rate: 5e-5, train batch size: 16
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_name "CIRCA_BERT_relaxed_e4_lr5e-5_b16" \
    --type false \
    --batch_size "16" \
    --epochs "4" \
    --learning_rate " 5e-5" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "6"

# Epochs: 2, Learning rate: 3e-5, train batch size: 16
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_name "CIRCA_BERT_relaxed_e2_lr3e-5_b16" \
    --type false \
    --batch_size "16" \
    --epochs "2" \
    --learning_rate " 3e-5" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "6"

# Epochs: 3, Learning rate: 3e-5, train batch size: 16
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_name "CIRCA_BERT_relaxed_e3_lr3e-5_b16" \
    --type false \
    --batch_size "16" \
    --epochs "3" \
    --learning_rate " 3e-5" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "6"

# Epochs: 4, Learning rate: 3e-5, train batch size: 16
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_name "CIRCA_BERT_relaxed_e4_lr3e-5_b16" \
    --type false \
    --batch_size "16" \
    --epochs "4" \
    --learning_rate " 3e-5" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "6"

# Epochs: 2, Learning rate: 2e-5, train batch size: 16
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_name "CIRCA_BERT_relaxed_e2_lr2e-5_b16" \
    --type false \
    --batch_size "16" \
    --epochs "2" \
    --learning_rate " 2e-5" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "6"

# Epochs: 3, Learning rate: 2e-5, train batch size: 16
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_name "CIRCA_BERT_relaxed_e3_lr2e-5_b16" \
    --type false \
    --batch_size "16" \
    --epochs "3" \
    --learning_rate " 2e-5" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "6"

# Epochs: 4, Learning rate: 2e-5, train batch size: 16
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_name "CIRCA_BERT_relaxed_e4_lr2e-5_b16" \
    --type false \
    --batch_size "16" \
    --epochs "4" \
    --learning_rate " 2e-5" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "6"

# Epochs: 2, Learning rate: 5e-5, train batch size: 32
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_name "CIRCA_BERT_relaxed_e2_lr5e-5_b32" \
    --type false \
    --batch_size "32" \
    --epochs "2" \
    --learning_rate " 5e-5" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "6"

# Epochs: 3, Learning rate: 5e-5, train batch size: 32
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_name "CIRCA_BERT_relaxed_e3_lr5e-5_b32" \
    --type false \
    --batch_size "32" \
    --epochs "3" \
    --learning_rate " 5e-5" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "6"

# Epochs: 4, Learning rate: 5e-5, train batch size: 32
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_name "CIRCA_BERT_relaxed_e4_lr5e-5_b32" \
    --type false \
    --batch_size "32" \
    --epochs "4" \
    --learning_rate " 5e-5" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "6"

# Epochs: 2, Learning rate: 3e-5, train batch size: 32
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_name "CIRCA_BERT_relaxed_e2_lr3e-5_b32" \
    --type false \
    --batch_size "32" \
    --epochs "2" \
    --learning_rate " 3e-5" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "6"

# Epochs: 3, Learning rate: 3e-5, train batch size: 32
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_name "CIRCA_BERT_relaxed_e3_lr3e-5_b32" \
    --type false \
    --batch_size "32" \
    --epochs "3" \
    --learning_rate " 3e-5" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "6"

# Epochs: 4, Learning rate: 3e-5, train batch size: 32
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_name "CIRCA_BERT_relaxed_e4_lr3e-5_b32" \
    --type false \
    --batch_size "32" \
    --epochs "4" \
    --learning_rate " 3e-5" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "6"

# Epochs: 2, Learning rate: 2e-5, train batch size: 32
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_name "CIRCA_BERT_relaxed_e2_lr2e-5_b32" \
    --type false \
    --batch_size "32" \
    --epochs "2" \
    --learning_rate " 2e-5" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "6"

# Epochs: 3, Learning rate: 2e-5, train batch size: 32
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_name "CIRCA_BERT_relaxed_e3_lr2e-5_b32" \
    --type false \
    --batch_size "32" \
    --epochs "3" \
    --learning_rate " 2e-5" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "6"

# Epochs: 4, Learning rate: 2e-5, train batch size: 32
python finetuning.py \
    --train_data "$BASE_DIR/data/circa-data-train.tsv" \
    --dev_data "$BASE_DIR/data/circa-data-dev.tsv" \
    --test_data "$BASE_DIR/data/circa-data-test.tsv" \
    --model_name "CIRCA_BERT_relaxed_e4_lr2e-5_b32" \
    --type false \
    --batch_size "32" \
    --epochs "4" \
    --learning_rate " 2e-5" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "$MODEL_TYPE" \
    --num_labels "9"

echo "Finished finetuning CIRCA with relaxed labels..."
