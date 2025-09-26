#!/bin/bash

# Hardcoded parameters (modify these values as needed)
MODEL="gpt-4o"
INPUT_REPORTS="../data/test_reports.csv"
INPUT_LABELS="../data/test_labels.csv" # input gt labels by default
GEN_DIR="../results/features/"
GT_FEATURES="../data/test_features.csv"

# Create output directory
mkdir -p $GEN_DIR

# Step 1: Run Inference
echo "Step 1: Starting feature inference process..."
python ./processor/vLLM.py \
  --model $MODEL \
  --input_reports $INPUT_REPORTS \
  --input_labels $INPUT_LABELS \
  --output_dir $GEN_DIR \


# Step 2: Run Evaluation
echo "Step 2: Starting evaluation process..."
python ./processor/eval.py \
  --gt_dir $GT_FEATURES \
  --gen_dir $GEN_DIR

echo "Feature extraction completed! Results saved to $GEN_DIR"
