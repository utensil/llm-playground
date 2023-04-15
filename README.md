# LLM Playground

This is a new era, with so many low-hanging fruits on the new magical tree, and so many new tools to reach them emerging every day. Things are changing so fast to play catch-up. I need a place to fix the settings and results of my experiments, this is it.

## Focus Matrix

I need to focus since I don't have unlimited time and resources.

It's a focus tensor actually: tools, models, datasets, data sources, hardware, metrics, hows.

- Tools: 
    - for inference and training, use text-generation-webui if applicable, and others otherwise.
        - focus on 4bit LoRA training with the biggest possible foundation models
        - also try some fully-fledged training on toy models and domain datasets
    - for creating, retrieving, and preprocessing datasets
- Models:
    - cover the mainstream ones with a focus on the ones that don't have legal issues for commercial use.
    - try to cover more types of LLM models to evaluate the differences.
- Datasets: 
    - focus on the high-quality ones that enhance coding, math, reasoning, and instruction-following capacities and have no legal issues for commercial use. 
    - natural languages include English, Chinese, and other languages I like.
    - programming languages include Python, C++, JavaScript, Rust, Lean 4, Julia, etc.
- Data sources:
    - the data source types that can be loaded with LlamaIndex/LangChain and injected into prompts.
    - other types I need.
- Hardware:
    - CPU-only: ~32 GB RAM
    - Consumer-grade GPU: RTX 3090Ti/A6000 ~48G VRAM
    - Cloud GPU: ~A100 80G VRAM
    - mostly use cloud services to switch between hardware easily
- Metrics:
    - light-weight metrics to evaluate models and datasets for general and domain-specific tasks
- Hows:
    - experiments to better understand why and how LLMs work

## The settings

- base on a bare-bone docker image like `nvidia/cuda:11.8.0-devel-ubuntu22.04` so I can switch between different cloud services easily
- these files are at the root or they could be folder-specific
    - `.env` for environment variables
    - `packages.txt` for apt packages
    - `requirements.txt` for Python packages
    - `datasets.txt` for datasets
    - `models.txt` for models
- the jupyter notebooks should
    - clone this repo and use helper scripts to fix other dependencies like text-generation-webui, GPTQ-for-LLaMa, alpaca_lora_4bit, etc.
    - use helper scripts to download datasets and models
    - upload results and models to HuggingFace for persistence
    - rely on credentials for Github, HuggingFace, etc.
