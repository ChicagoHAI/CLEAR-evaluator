#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="${SCRIPT_DIR}"
PYTHON_BIN="${PYTHON_BIN:-python}"

usage() {
  cat <<'USAGE'
Usage:
  run.bash --gen-reports PATH --label-model NAME --feature-model NAME [options]

Required arguments:
  --gen-reports PATH         CSV of generated reports with columns: study_id, report
  --label-model NAME         Model identifier understood by label configs
  --feature-model NAME       Model identifier understood by feature configs

Optional arguments:
  --gt-reports PATH          CSV of ground-truth/reference reports
  --label-backbone {azure|vllm}
  --feature-backbone {azure|vllm}
  --output-root PATH         Directory for pipeline outputs (default: runs/<timestamp>)
  --feature-eval-enable-llm  Enable LLM metric during feature evaluation
  --feature-eval-scoring-llm NAME  LLM model used for scoring (implies --feature-eval-enable-llm)
  --python PATH              Python interpreter to use (default: $PYTHON_BIN)
  -h, --help                 Show this help message and exit

Examples:
  ./run.bash --gen-reports data/generated.csv \\
            --gt-reports data/reference.csv \\
            --label-model gpt-4o --feature-model gpt-4o
USAGE
}

GEN_REPORTS="${ROOT_DIR}/data/test_reports.csv"
GT_REPORTS="${ROOT_DIR}/data/test_reports.csv"
LABEL_BACKBONE="azure"
LABEL_MODEL="gpt-4o"
FEATURE_BACKBONE="azure"
FEATURE_MODEL="gpt-4o"
OUTPUT_ROOT="${ROOT_DIR}/runs/$(date +"%Y%m%d_%H%M%S")"
ENABLE_LLM=true
SCORING_LLM="gpt-4o-mini"

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --gen-reports)
      [[ $# -ge 2 ]] || { echo "error: --gen-reports requires a value" >&2; exit 1; }
      GEN_REPORTS="$2"
      shift 2
      ;;
    --gt-reports)
      [[ $# -ge 2 ]] || { echo "error: --gt-reports requires a value" >&2; exit 1; }
      GT_REPORTS="$2"
      shift 2
      ;;
    --label-backbone)
      [[ $# -ge 2 ]] || { echo "error: --label-backbone requires a value" >&2; exit 1; }
      LABEL_BACKBONE="$2"
      shift 2
      ;;
    --label-model)
      [[ $# -ge 2 ]] || { echo "error: --label-model requires a value" >&2; exit 1; }
      LABEL_MODEL="$2"
      shift 2
      ;;
    --feature-backbone)
      [[ $# -ge 2 ]] || { echo "error: --feature-backbone requires a value" >&2; exit 1; }
      FEATURE_BACKBONE="$2"
      shift 2
      ;;
    --feature-model)
      [[ $# -ge 2 ]] || { echo "error: --feature-model requires a value" >&2; exit 1; }
      FEATURE_MODEL="$2"
      shift 2
      ;;
    --output-root)
      [[ $# -ge 2 ]] || { echo "error: --output-root requires a value" >&2; exit 1; }
      OUTPUT_ROOT="$2"
      shift 2
      ;;
    --feature-eval-enable-llm)
      ENABLE_LLM=true
      shift
      ;;
    --feature-eval-scoring-llm)
      [[ $# -ge 2 ]] || { echo "error: --feature-eval-scoring-llm requires a value" >&2; exit 1; }
      SCORING_LLM="$2"
      ENABLE_LLM=true
      shift 2
      ;;
    --python)
      [[ $# -ge 2 ]] || { echo "error: --python requires a value" >&2; exit 1; }
      PYTHON_BIN="$2"
      shift 2
      ;;
    *)
      echo "error: unknown option '$1'" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$GEN_REPORTS" ]]; then
  echo "error: --gen-reports is required" >&2
  usage >&2
  exit 1
fi
if [[ -z "$LABEL_MODEL" ]]; then
  echo "error: --label-model is required" >&2
  usage >&2
  exit 1
fi
if [[ -z "$FEATURE_MODEL" ]]; then
  echo "error: --feature-model is required" >&2
  usage >&2
  exit 1
fi

if [[ ! -f "$GEN_REPORTS" ]]; then
  echo "error: generated reports file not found: $GEN_REPORTS" >&2
  exit 1
fi
if [[ -n "$GT_REPORTS" && ! -f "$GT_REPORTS" ]]; then
  echo "error: ground-truth reports file not found: $GT_REPORTS" >&2
  exit 1
fi

if [[ "$LABEL_BACKBONE" != "azure" && "$LABEL_BACKBONE" != "vllm" ]]; then
  echo "error: --label-backbone must be 'azure' or 'vllm'" >&2
  exit 1
fi
if [[ "$FEATURE_BACKBONE" != "azure" && "$FEATURE_BACKBONE" != "vllm" ]]; then
  echo "error: --feature-backbone must be 'azure' or 'vllm'" >&2
  exit 1
fi

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${ROOT_DIR}"
fi

CMD=("${PYTHON_BIN}" "${ROOT_DIR}/main.py"
  "--gen-reports" "${GEN_REPORTS}"
  "--label-backbone" "${LABEL_BACKBONE}"
  "--label-model" "${LABEL_MODEL}"
  "--feature-backbone" "${FEATURE_BACKBONE}"
  "--feature-model" "${FEATURE_MODEL}"
  "--output-root" "${OUTPUT_ROOT}")

if [[ -n "$GT_REPORTS" ]]; then
  CMD+=("--gt-reports" "${GT_REPORTS}")
fi
if [[ "$ENABLE_LLM" == true ]]; then
  CMD+=("--feature-eval-enable-llm")
fi
if [[ -n "$SCORING_LLM" ]]; then
  CMD+=("--feature-eval-scoring-llm" "${SCORING_LLM}")
fi

printf 'Running CLEAR pipeline from %s\n' "$ROOT_DIR"
printf 'Command:'
printf ' %q' "${CMD[@]}"
printf '\n'

exec "${CMD[@]}"
