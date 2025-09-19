# CLEAR: A Clinically-Grounded Tabular Framework for Radiology Report Evaluation

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


## Usage

Example scripts are available at `./label/run_label.bash` and `./feature/run_feature.bash`. Please use them as templates and adjust the variables to fit your configuration.

