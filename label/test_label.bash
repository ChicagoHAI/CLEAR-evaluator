#!/bin/bash

set -euo pipefail
trap 'echo "run_label.bash failed at line $LINENO"; exit 1' ERR

# Hardcoded parameters (modify these values as needed)
MODEL="llama-3.1-8b-sft"
BACKBONE="vllm"
INPUT_REPORTS="../data/test_reports.csv"
GEN_DIR="../results/labels/"
GT_LABELS="../data/test_labels.csv"
SKIP_INFERENCE="false"

# Create output directory
mkdir -p "$GEN_DIR"

# Step 1: Run inference
echo "Starting inference process..."
if [ "$SKIP_INFERENCE" != "true" ]; then
  echo "Running inference (SKIP_INFERENCE=$SKIP_INFERENCE)."
  if [ "$BACKBONE" == "azure" ]; then
    echo "Using Azure backbone."
    python ./processor/AzureOpenAI.py \
      --model_name $MODEL \
      --reports "$INPUT_REPORTS" \
      --output "$GEN_DIR"
  elif [ "$BACKBONE" == "vllm" ]; then
    echo "Using vLLM backbone."
    python ./processor/vLLM.py \
      --model_name $MODEL \
      --reports "$INPUT_REPORTS" \
      --output "$GEN_DIR"
  else
    echo "Unsupported backbone: $BACKBONE"
    exit 1
  fi
else
  echo "Skipping inference step (SKIP_INFERENCE=true)."
fi

# Step 2: Run evaluation
echo "Starting evaluation process..."
python ./processor/eval.py \
  --gt_dir "$GT_LABELS" \
  --gen_dir "$GEN_DIR" \
  --model_name "$MODEL"

echo "Label extraction completed! Results saved to $GEN_DIR"
