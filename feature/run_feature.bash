#!/bin/bash

# Hardcoded parameters (modify these values as needed)
MODEL="llama-3.1-8b"
INPUT_CSV="/path/to/your/reports.csv"
OUTPUT_DIR="/path/to/your/label_output"
GT_DIR="/path/to/your/ground_truth"

# Create output directory
mkdir -p $OUTPUT_DIR

# Step 1: Label Inference
echo "Step 1: Starting label inference process..."
python ./processor/vLLM.py \
  --model $MODEL \
  --input_csv $INPUT_CSV \
  --o $OUTPUT_DIR \
  --prompt $PROMPT_VERSION


# Step 2: Evaluation
echo "Step 2: Starting evaluation process..."
python ./processor/eval.py \
  --gt_dir $GT_DIR \
  --gen_dir $OUTPUT_DIR


echo "Complete pipeline execution finished!"
