# config.py

MODEL_CONFIGS = {
    "gpt-4o": {
        "api_key": "", # API key for Azure API.
        "api_version": "2025-01-01-preview", # Endpoint for Azure API.
        "endpoint": "", # Endpoint for Azure API.
        "deployment": "gpt-4o", # Deployment name for Azure API.
        "max_tokens": 4096
    },
    "llama-3.1-8b": {
        "model_path": "meta-llama/Llama-3.1-8B-Instruct",
        "temperature": 1e-5,
        "max_tokens": 4096,
        "tensor_parallel_size": 4 # 4*A100
    },
     "o1-mini": {
        "api_key": "", # API key for Azure API.
        "api_version": "2024-08-01-preview", # Endpoint for Azure API.
        "endpoint": "", # Endpoint for Azure API.
        "deployment": "o1-mini" # Deployment name for Azure API.
    }
}