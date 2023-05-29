# Assuming image: winglian/axolotl-runpod:main-cu118-2.0.0
set -euxo pipefail

set -x

WORKSPACE=${WORKSPACE:-"/workspace"}

cd $WORKSPACE/llm-playground/notebooks/
pip install -r requirements.txt