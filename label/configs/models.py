# config.py

MODEL_CONFIGS = {
    "gpt-4o": {
        "api_key": "", # API key for Azure API.
        "api_version": "2025-01-01-preview", # Endpoint for Azure API.
        "endpoint": "", # Endpoint for Azure API.
        "deployment": "gpt-4o", # Deployment name for Azure API.
        "max_tokens": 4096
    },
    "llama-3.1-8b-sft": {
        "model_path": "/net/projects/chacha/hpo_models/best/llama3.1-full-sft-v24",
        "temperature": 1e-5,
        "max_tokens": 4096,
        "tensor_parallel_size": 4 # 4*A100
    }
}
