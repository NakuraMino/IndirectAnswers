#!/bin/bash

## File for testing on the unmatched, relaxed dataset

BASE_DIR="/home/azureuser/IndirectAnswers"  # IndirectAnswers directory
UNMATCHED_DIR="$BASE_DIR/data/CIRCA/unmatched/scenario"

cd "$BASE_DIR"

Modes=('q' 'a' 'qa') # q=question, a=answer
LearningRates=('3e-5' '2e-5' '3e-5')
ScenarioNum="0 1 2 3 4 5 6 7 8 9"

echo "Testing on CIRCA with unmatched, relaxed labels..."

for i in ${!Modes[@]}; do
    for j in $ScenarioNum; do # 10 iterations, leaving a different scenario out each time
        mode=${Modes[i]}
        lr=${LearningRates[i]}
        echo "STARTING... Mode: $mode, Learning rate: $lr, Scenario: $j"
        # Evaluate on test dataset
        python testing.py \
            --test_data "${UNMATCHED_DIR}${j}/circa-data-test.tsv" \
            --model_type "$BASE_DIR/models/CIRCA_BERT_unmatched_relaxed_${mode}_s${j}_e3_lr${lr}_b32" \
            --dataset_type "CIRCA" \
            --batch_size "32" \
            --num_labels "6"
            --output_path "$BASE_DIR/results/CIRCA-test-results.tsv"
        
        # Evaluate on dev dataset
        python testing.py \
            --test_data "${UNMATCHED_DIR}${j}/circa-data-dev.tsv" \
            --model_type "$BASE_DIR/models/CIRCA_BERT_unmatched_relaxed_${mode}_s${j}_e3_lr${lr}_b32" \
            --dataset_type "CIRCA" \
            --batch_size "32" \
            --num_labels "6"
            --output_path "$BASE_DIR/results/CIRCA-dev-results.tsv"
    done
done

echo "Finished testing CIRCA with unmatched, relaxed labels..."
