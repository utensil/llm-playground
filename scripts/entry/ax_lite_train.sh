#!/bin/bash
#
# Container source: https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/docker/Dockerfile-runpod
#
#
# To run this in RunPod with `winglian/axolotl-runpod:main-py3.10-cu118-2.0.1`, set
# Expose HTTP Ports (Max 10): 7860,8888
# docker command: `bash -c "curl -H 'Cache-Control: no-cache' https://raw.githubusercontent.com/utensil/llm-playground/main/scripts/entry/ax_lite.sh -sSf | bash"`
# JUPYTER_PASSWORD change to your secret
# HUGGINGFACE_TOKEN change to your token from https://huggingface.co/settings/tokens
# WORKSPACE /workspace/
# WANDB_API_KEY change to your key from https://wandb.ai/authorize
#
# MUST mount volume disk at /content/
#
# To test this in Codespaces, run `cd /workspaces/ && WORKSPACE=/workspaces/ llm-playground/scripts/entry/ax_lite.sh`

set -euxo pipefail

set -x

WORKSPACE=${WORKSPACE:-"/workspace"}

export DEBIAN_FRONTEND=noninteractive

# make the cache live on volume disk
rm -rf /root/.cache
mkdir -p /content/cache
ln -s /content/cache /root/.cache

# prepare jupyter
pip install jupyterhub notebook jupyterlab jupyterlab-git ipywidgets

# prepare monitoring GPU
pip install nvitop

AXOLOTL_GIT=${AXOLOTL_GIT:-"https://github.com/OpenAccess-AI-Collective/axolotl"}
AXOLOTL_GIT_BRANCH=${AXOLOTL_GIT_BRANCH:-"main"}

# update axolotl
cd $WORKSPACE
if [ ! -d "axolotl-update" ]; then
  git clone -b $AXOLOTL_GIT_BRANCH --single-branch $AXOLOTL_GIT axolotl-update
# don't update between runs yet
# else
#   cd axolotl-update && git pull && cd ..
fi
cp -rf axolotl-update axolotl
cd axolotl
pip install -e .

mkdir -p /content/
cd /content/
if [ ! -d "llm-playground" ]; then
  git clone https://github.com/utensil/llm-playground
# don't update between run yet
# else
#   cd llm-playground && git pull && cd ..
fi

cd $WORKSPACE

# don't update peft
# PEFT_COMMIT_HASH=${PEFT_COMMIT_HASH:-"main"}
# pip install git+https://github.com/huggingface/peft.git@$PEFT_COMMIT_HASH

JUPYTER_PASSWORD=${JUPYTER_PASSWORD:-"axolotl"}

echo "Launching Jupyter Lab with nohup..."
cd /
nohup jupyter lab --allow-root --no-browser --port=8888 --ip=* --ServerApp.token=$JUPYTER_PASSWORD --ServerApp.allow_origin=* --ServerApp.preferred_dir=$WORKSPACE &

TRAINING_CONFIG=${TRAINING_CONFIG:-""}

if [[ -n $TRAINING_CONFIG ]]
then
  # Fixes update hang
  # apt-get update
  apt-get install -y aria2
  git lfs install
  pip install requests huggingface_hub discord.py pexpect pytest
  # Fixes https://github.com/huggingface/accelerate/pull/1648#issuecomment-1610178618
  # pip install --upgrade --force-reinstall git+https://github.com/huggingface/accelerate.git
  pip install -U git+https://github.com/huggingface/peft.git
  pip install git+https://github.com/utensil/runpod-python@pod
  # Fixes https://github.com/microsoft/DeepSpeed/issues/3963
  pip install pydantic==1.10.10
  # Optional: for deepspeed
  pip3 install -U torch --index-url https://download.pytorch.org/whl/cu118
  cd /content/llm-playground
  (python /content/llm-playground/helper/download-model.py $PREDOWNLOAD_MODEL || true )
  (accelerate launch helper/train.py $TRAINING_CONFIG || python /content/llm-playground/helper/terminate-runpod.py )
fi

sleep infinity
