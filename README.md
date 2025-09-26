# CLEAR: A Clinically Grounded Tabular Framework for Radiology Report Evaluation

This is the codebase for an end-to-end evaluator for radiology report evaluation based on taxonomy first proposed in [CLEAR](https://arxiv.org/abs/2505.16325) (2025 EMNLP Findings).

## CLEAR Framework
![CLEAR overview](pics/CLEAR_overview.png)

## Codebase Structure
```
.
├── environment.yaml
├── feature
│   ├── configs
│   │   ├── models.py
│   │   └── prompts.py
│   ├── processor
│   │   ├── AzureOpenAI.py
│   │   ├── eval.py
│   │   └── vLLM.py
│   └── run_feature.bash
├── label
│   ├── configs
│   │   ├── models.py
│   │   └── prompts.py
│   ├── processor
│   │   ├── AzureOpenAI.py
│   │   ├── eval.py
│   │   └── vLLM.py
│   └── run_label.bash
├── models
│   ├── README.md
│   └── train
│       └── LLaMA-Factory
├── README.md
└── requirements.txt
```

## Installation

In the main README, we would only demonstrate how to use CLEAR evaluator. If interested in post-training for a specialized local small-size evaluator, please refer /models for more details.

We recommend using conda for environment management. Please run the following command for setting up environment.

```bash
conda env create -f environment.yaml
pip install -r requirements.txt
```

## Component

The evaluator runs in two sequential modules. The table below captures what each module is responsible for and how to configure it.

| Aspect | Module 1: Label Extraction | Module 2: Description Extraction |
| --- | --- | --- |
| Function | Predicts the 13 CLEAR condition labels (`positive`, `negative`, `unclear`) from full radiology reports using a 5-shot JSON-tagged prompt. | Expands positive findings into structured features (First Occurrence, Change, Severity, Urgency, Descriptive Location, Action/Recommendation) using radiologist-curated templates. |
| Supported open-source model requirements (vLLM) | `model_path`, `temperature`, `max_tokens`, `tensor_parallel_size` (see `label/configs/models.py`). | `model_path`, `temperature`, `max_tokens`, `tensor_parallel_size` (see `feature/configs/models.py`). |
| Supported closed-source model requirements (AzureOpenAI) | `api_key`, `api_version`, `endpoint`, `deployment`, optional `max_tokens` (see `label/configs/models.py`). | `api_key`, `api_version`, `endpoint`, `deployment`, optional `max_tokens` (see `feature/configs/models.py`). |
| Input file | Reports CSV with `study_id` and `report`; evaluation expects ground-truth labels CSV with the 13 CLEAR condition columns. | Reports CSV plus label CSV containing CLEAR condition columns (used to identify positive conditions); evaluation consumes ground-truth feature JSON/CSV. |
| Prompting | `label/configs/prompts.py` provides the system prompt with five illustrative exemplars covering all conditions and enforced JSON schema. | `feature/configs/prompts.py` generates per-condition prompts; templates adapt to each condition and feature type before inference. |
| Intermediate output file | Predictions saved to `GEN_DIR/tmp/output_labels_<model>.json`; evaluation metrics written to `GEN_DIR/label_metrics_<model>.csv`. | Feature JSON saved to `GEN_DIR/output_feature_<model>.json`; evaluation exports `results_qa_avg.csv` and `results_ie_avg.csv` in `GEN_DIR`. |
| Scoring | `processor/eval.py` reports `Pos F1`, `Pos F1_5`, `Pos micro F1`, `Neg F1`, `Neg F1_5`, `Neg micro F1`, plus per-condition positive/negative F1 scores. | QA features (First Occurrence, Change, Severity) score `Acc. (micro)`, `Acc. (macro)`, `F1 (micro)`, `F1 (macro)`; IE features (Descriptive Location, Recommendation) score `o1-mini score`, `ROUGE-L`, `BLEU-4`. |

## Materials

1. We release **CLEAR-Bench**, our adaptable expert evaluation dataset designed for use with the CLEAR evaluator, on [PhysioNet](https://physionet.org/). *(Coming very soon!)*

2. To accelerate open-source model inference, we implement our backend using the **vLLM** architecture. For more details, please refer to the [official vLLM documentation](https://docs.vllm.ai/en/latest/).

3. To address privacy concerns when working with medical data, we follow the guidelines outlined in [Responsible Use of MIMIC Data with Online Services like GPT](https://physionet.org/news/post/gpt-responsible-use). Specifically, we utilize the [Azure OpenAI Service](https://azure.microsoft.com/en-us/products/ai-foundry/models/openai/) to enable secure use of commercial, closed-source models.


## Usage Tips

1. Example scripts are available at `./label/run_label.bash` and `./feature/run_feature.bash`. Please use them as templates and adjust the variables to fit your configuration.

2. model name in bash must exist in configs/model.py

3. skip inference by setting SKIP_INFERENCE=True

4. labeling schema in evaluation and dataset using 0 (negative), 1 (postive), -1 (unclear)

5. ensure your input reports has a column named ['report'] containing both FINDINGS and IMPRESSION

6. make sure the model you pass under the vLLM backbone always have temperature, max_tokens, and tensor_parallel_size features
