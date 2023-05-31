# Assuming image: winglian/axolotl-runpod:main-cu118-2.0.0

# TODO: maybe no -U or make it optional
pip install bitsandbytes
pip install git+https://github.com/huggingface/transformers.git 
pip install git+https://github.com/huggingface/peft.git
pip install git+https://github.com/huggingface/accelerate.git

# Need to change this before uncommenting
# export WANDB_PROJECT=runpod

# Might need to fix bitsandbytes
# WORKSPACE=${WORKSPACE:-"/workspace"}
# cd $WORKSPACE
# git clone https://github.com/TimDettmers/bitsandbytes
# cd bitsandbytes/
# CUDA_VERSION=118 make cuda11x
# python setup.py install
# python -m bitsandbytes

