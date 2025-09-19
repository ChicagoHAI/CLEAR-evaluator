#!/bin/bash

# Hardcoded parameters (modify these values as needed)
MODEL="llama-3.1-8b"
INPUT_CSV="/path/to/your/input.csv"
OUTPUT_DIR="/path/to/your/output"
GT_DIR="/path/to/your/ground_truth"
PROMPT_VERSION="5"

# Create output directory
mkdir -p $OUTPUT_DIR

# Step 1: Run inference
echo "Starting inference process..."
python ./processor/vLLM.py \
  --model $MODEL \
  --input_csv $INPUT_CSV \
  --o $OUTPUT_DIR \
  --prompt $PROMPT_VERSION

# Step 2: Run evaluation
echo "Starting evaluation process..."
python ./processor/eval.py \
  --gt_dir $GT_DIR \
  --gen_dir $OUTPUT_DIR \
  --prompt $PROMPT_VERSION

echo "Process completed! Results saved to $OUTPUT_DIR"