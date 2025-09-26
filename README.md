# CLEAR: A Clinically Grounded Tabular Framework for Radiology Report Evaluation

This is the implementation of CLEAR: A Clinically-Grounded Tabular Framework for
Radiology Report Evaluation. 

## Project Structure
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

We recommend using conda for environment management.

Firstly, please install LLaMA-Factory as a submodule.

Secondly, please run the following command for setting up environment.

```bash
conda env create -f environment.yaml
pip install -r requirements.txt
```

## Component

1. Module 1: Label Extraction Module (5-shot prompting)


## Usage

1. Example scripts are available at `./label/run_label.bash` and `./feature/run_feature.bash`. Please use them as templates and adjust the variables to fit your configuration.

2. model name in bash must exist in configs/model.py

3. skip inference by setting SKIP_INFERENCE=True

4. labeling schema in evaluation and dataset using 0, 1, -1

5. ensure your input reports has a column named ['report'] containing both FINDINGS and IMPRESSION

6. make sure the model you pass under the vLLM backbone always have temperature, max_tokens, and tensor_parallel_size features

