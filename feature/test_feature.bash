#!/bin/bash

# Script configuration
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Hardcoded parameters (modify these values as needed)
MODEL="gpt-4o"
BACKBONE="azure"
INPUT_REPORTS="${ROOT_DIR}/data/test_reports.csv"
INPUT_LABELS="${ROOT_DIR}/data/test_labels.csv" # input gt labels by default
GEN_DIR="${ROOT_DIR}/results/features"
GT_FEATURES="${ROOT_DIR}/data/test_features.json"
GEN_FILE="${GEN_DIR}/tmp/output_feature_${MODEL}.json"
SKIP_INFERENCE="false"

# Create output directory
mkdir -p "${GEN_DIR}/tmp"

# Step 1: Run Inference
echo "Step 1: Starting inference process..."
if [ "$SKIP_INFERENCE" != "true" ]; then
  echo "Running inference (SKIP_INFERENCE=$SKIP_INFERENCE)."
  if [ "$BACKBONE" == "azure" ]; then
    echo "Using Azure backbone."
    python "${SCRIPT_DIR}/processor/AzureOpenAI.py" \
      --model "${MODEL}" \
      --reports "${INPUT_REPORTS}" \
      --labels "${INPUT_LABELS}" \
      --output "${GEN_DIR}"
  elif [ "$BACKBONE" == "vllm" ]; then
    echo "Using vLLM backbone."
    python "${SCRIPT_DIR}/processor/vLLM.py" \
      --model_name "${MODEL}" \
      --reports "${INPUT_REPORTS}" \
      --labels "${INPUT_LABELS}" \
      --output "${GEN_DIR}"
  else
    echo "Unsupported backbone: $BACKBONE"
    exit 1
  fi
else
  echo "Skipping inference step (SKIP_INFERENCE=true)."
fi

# Step 2: Run Evaluation
echo "Step 2: Starting evaluation process..."
python "${SCRIPT_DIR}/processor/eval.py" \
  --gt_path "${GT_FEATURES}" \
  --gen_path "${GEN_FILE}" \
  --model_name "${MODEL}" \
  --metric_path "${GEN_DIR}" \
  --enable_llm_metric \
  --scoring_llm "gpt-4o-mini"

echo "Feature extraction completed! Results saved to ${GEN_DIR}"
