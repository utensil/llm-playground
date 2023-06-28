import os
import sys
from pathlib import Path
import fire
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
axolotl_root = os.getenv("AXOLOTL_ROOT", os.path.abspath(os.path.join(project_root, "../axolotl")))
src_dir = os.path.join(axolotl_root, "src")
scripts_dir = os.path.join(axolotl_root, "scripts")
sys.path.insert(0, src_dir)
sys.path.insert(0, scripts_dir)

from huggingface_hub import login
login(os.environ.get("HUGGINGFACE_TOKEN"), add_to_git_credential=True)

import finetune
from axolotl.utils.trainer import setup_trainer as setup_trainer_orig

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

def train_ex(
    config: Path = Path("configs/"),
    prepare_ds_only: bool = False,
    **kwargs,
):
  logging.info('train_ex before')
  finetune.train(config, prepare_ds_only, **kwargs)
  logging.info('train_ex after')

def setup_trainer_ex(cfg, train_dataset, eval_dataset, model, tokenizer):
  logging.info('setup_trainer_ex before')
  logging.info(f'cfg.some_config = {cfg.some_config}')
  trainer = setup_trainer_orig(cfg, train_dataset, eval_dataset, model, tokenizer)
  logging.info('setup_trainer_ex after')
  return trainer

finetune.setup_trainer = setup_trainer_ex

if __name__ == "__main__":
    fire.Fire(train_ex)

    