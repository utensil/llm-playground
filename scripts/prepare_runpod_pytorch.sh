#!/bin/bash
# 
# Container source: https://github.com/huggingface/transformers/blob/main/docker/transformers-pytorch-deepspeed-latest-gpu/
#
# This script:
# - cleaned up `prepare_runpod.sh`
# - adapted https://github.com/winglian/axolotl/blob/main/scripts/setup-runpod.sh
# - adapted https://github.com/runpod/containers/tree/main/oobabooga
#
# To run this in RunPod with `huggingface/transformers-pytorch-deepspeed-latest-gpu`, set
# Expose HTTP Ports (Max 10): 7860,8888
# docker command: `bash -c "curl -H 'Cache-Control: no-cache' https://raw.githubusercontent.com/utensil/llm-playground/main/scripts/prepare_runpod_pytorch.sh -sSf | bash"`
# WEBUI chatbot
# JUPYTER_PASSWORD change to your secret
# HUGGINGFACE_TOKEN change to your token from https://huggingface.co/settings/tokens
# SUDO nosudo
# WORKSPACE /workspace/
#
# To test this in Codespaces, run `cd /workspaces/ && WORKSPACE=/workspaces/ llm-playground/scripts/prepare_runpod_pytorch.sh`

set -euxo pipefail

set -x

CODESPACES=${CODESPACES:-""}
WORKSPACE=${WORKSPACE:-"/workspace"}

cd $WORKSPACE

if [ -z "$CODESPACES" ]; then
    nvidia-smi
    num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    gpu_indices=$(seq 0 $((num_gpus - 1)) | paste -sd "," -)
    export CUDA_VISIBLE_DEVICES=$gpu_indices
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
fi

cd $WORKSPACE

if [ ! -d "llm-playground" ]; then
  git clone https://github.com/utensil/llm-playground
fi

cd llm-playground

export DEBIAN_FRONTEND=noninteractive
./helper/prepare.sh

# Skip CUDA in Codespaces
if [ -z "$CODESPACES" ]; then
    pip3 install --force-reinstall https://download.pytorch.org/whl/nightly/cu117/torch-2.0.0.dev20230301%2Bcu117-cp38-cp38-linux_x86_64.whl --index-url https://download.pytorch.org/whl/nightly/cu117
    if [ -z "${TORCH_CUDA_ARCH_LIST}" ]; then # only set this if not set yet
        # this covers most common GPUs that the installed version of pytorch supports
        # python -c "import torch; print(torch.cuda.get_arch_list())"
        export TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6+PTX"
    fi
fi

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

python ./helper/upload.py

cd notebooks
pip3 install -r requirements.txt

cd $WORKSPACE

if [ ! -d "text-generation-webui" ]; then
  git clone https://github.com/oobabooga/text-generation-webui
fi

cd $WORKSPACE/text-generation-webui/

git pull

pip3 install -r requirements.txt

TMP=$WORKSPACE/tmp/
rm -rf $TMP
mkdir -p $TMP

mv models $TMP
mv loras $TMP
mv training/datasets $TMP

ln -s $WORKSPACE/llm-playground/models ./models
ln -s $WORKSPACE/llm-playground/loras ./loras
ln -s $WORKSPACE/llm-playground/datasets ./training/datasets

cp $WORKSPACE/llm-playground/storage/3b.txt ./training/datasets/3b.txt

JUPYTER_PASSWORD=${JUPYTER_PASSWORD:-"pytorch"}

if [[ $JUPYTER_PASSWORD ]]
then
  echo "Launching Jupyter Lab"
  cd $WORKSPACE
  jupyter nbextension enable --py widgetsnbextension
  # https://stackoverflow.com/a/63372590/200764
  rm -rf `jupyter --config-dir`
  nohup jupyter lab --allow-root --no-browser --port=8888 --ip=* --ServerApp.token=$JUPYTER_PASSWORD --ServerApp.allow_origin=* --ServerApp.preferred_dir=$WORKSPACE &
fi

sleep infinity