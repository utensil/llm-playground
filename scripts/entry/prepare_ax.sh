#!/bin/bash
#
# Container source: https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docker/Dockerfile-runpod
#
#
# To run this in RunPod with `winglian/axolotl-runpod:main-cu118-2.0.0`, set
# Expose HTTP Ports (Max 10): 7860,8888
# docker command: `bash -c "curl -H 'Cache-Control: no-cache' https://raw.githubusercontent.com/utensil/llm-playground/main/scripts/entry/prepare_ax.sh -sSf | bash"`
# JUPYTER_PASSWORD change to your secret
# HUGGINGFACE_TOKEN change to your token from https://huggingface.co/settings/tokens
# SUDO nosudo
# WORKSPACE /workspace/
# WANDB_API_KEY change to your key from https://wandb.ai/authorize
#
# To test this in Codespaces, run `cd /workspaces/ && WORKSPACE=/workspaces/ llm-playground/scripts/entry/prepare_ax.sh`

set -euxo pipefail

set -x

WORKSPACE=${WORKSPACE:-"/workspace"}

cd $WORKSPACE

if [ ! -d "llm-playground" ]; then
  git clone https://github.com/utensil/llm-playground
fi

cd llm-playground

export DEBIAN_FRONTEND=noninteractive
./helper/prepare.sh

cd $WORKSPACE/llm-playground

LOAD_MODEL=${LOAD_MODEL:-""}
LOAD_DATASET=${LOAD_DATASET:-""}

if [ ! -z "$LOAD_MODEL" ]; then
    python ./helper/download-model.py $LOAD_MODEL
fi

if [ ! -z "$LOAD_DATASET" ]; then
    python ./helper/download-dataset.py $LOAD_DATASET
fi

mkdir -p $WORKSPACE/llm-playground/models
mkdir -p $WORKSPACE/llm-playground/loras
mkdir -p $WORKSPACE/llm-playground/datasets

python ./helper/storage.py

bash $WORKSPACE/llm-playground/scripts/prepare_jupyter.sh
bash $WORKSPACE/llm-playground/scripts/prepare_training.sh
bash $WORKSPACE/llm-playground/scripts/prepare_qlora.sh

cp -r $WORKSPACE/llm-playground/notebooks/axolotl/config/* $WORKSPACE/axolotl/examples/

JUPYTER_PASSWORD=${JUPYTER_PASSWORD:-"pytorch"}

bash $WORKSPACE/llm-playground/scripts/nohup_jupyter.sh

sleep infinity