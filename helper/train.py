import os
import sys
from pathlib import Path
import fire
import logging
import wandb
import pandas as pd
from discord import SyncWebhook
from addict import Dict
import yaml
import runpod
from transformers.trainer_callback import TrainerCallback

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
from axolotl.utils.dict import DictDefault

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# TODO: avoid code dup
def notify_discord(msg):
    webhook = SyncWebhook.from_url(os.getenv("DISCORD_WEBHOOK_URL"))
    webhook.send(msg)

def log_info(msg):
    logging.info(msg)
    notify_discord(msg)

def log_error(msg, exc_info=None):
    logging.error(msg, exc_info=exc_info)
    if exc_info is not None:
        notify_discord(f'{msg}: {exc_info}')
    else:
        notify_discord(msg)

def train_ex(
    config,
    prepare_ds_only: bool = False,
    **kwargs,
):
    logging.info('train_ex before')

    config = Path(config.strip())
    log_info(f"Prepare training with config: {config}")

    # TODO: avoid code dup
    # load the config from the yaml file
    # Mostly borrowed from https://github.com/utensil/axolotl/blob/local_dataset/scripts/finetune.py
    with open(config, encoding="utf-8") as file:
        cfg: DictDefault = DictDefault(yaml.safe_load(file))

        # if there are any options passed in the cli, if it is something that seems valid from the yaml,
        # then overwrite the value
        cfg_keys = cfg.keys()
        for k, _ in kwargs.items():
            # if not strict, allow writing to cfg even if it's not in the yml already
            if k in cfg_keys or not cfg.strict:
                # handle booleans
                if isinstance(cfg[k], bool):
                    cfg[k] = bool(kwargs[k])
                else:
                    cfg[k] = kwargs[k]

    # os.environ["WANDB_RESUME"] = "auto"
    if cfg.wandb_project is not None:
        run_id = cfg.wandb_run_id or wandb.util.generate_id()
        run = wandb.init(project=cfg.wandb_project, id=run_id) #, resume=True)
        os.environ["WANDB_RUN_ID"] = run_id

    finetune.train(config, prepare_ds_only, **kwargs)

    if cfg.runpod.one_shot:        
        runpod.api_key = os.getenv("RUNPOD_API_KEY")

        pod_id = os.getenv("RUNPOD_POD_ID")

        runpod.terminate_pod(pod_id)

        log_info(f"Pod {pod_id} terminated on train end")

    logging.info('train_ex after')

def log_eval_prediction(ep):
    data = {
        'input': ep.inputs,
        'prediction': ep.predictions,
        'label_id': ep.label_ids
    }
    logging.info(data)
    # df = pd.DataFrame(data)
    # table = wandb.Table(dataframe=df)
    # wandb.run.log({"eval_entries": my_table})

class OneshotCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        pass
        # logging.info('OneshotCallback on_train_begin')
        
        # runpod.api_key = os.getenv("RUNPOD_API_KEY")

        # pod_id = os.getenv("RUNPOD_POD_ID")

        # runpod.terminate_pod(pod_id)

        # log_info(f"Pod {pod_id} terminated on train begin")

    def on_train_end(self, args, state, control, **kwargs):
        logging.info('OneshotCallback on_train_end')
        
        runpod.api_key = os.getenv("RUNPOD_API_KEY")

        pod_id = os.getenv("RUNPOD_POD_ID")

        runpod.terminate_pod(pod_id)

        log_info(f"Pod {pod_id} terminated on train end")

def setup_trainer_ex(cfg, train_dataset, eval_dataset, model, tokenizer):
    logging.info('setup_trainer_ex before')
    logging.info(f'cfg.runpod.one_shot = {cfg.runpod.one_shot}')
    trainer = setup_trainer_orig(cfg, train_dataset, eval_dataset, model, tokenizer)
    trainer.args.include_inputs_for_metrics = True
    compute_metrics_orig = trainer.compute_metrics

    def compute_metrics(ep):
        metrics = compute_metrics_orig(ep) if compute_metrics_orig else {}
        log_eval_prediction(ep)
        return metrics

    trainer.compute_metrics = compute_metrics

    # if cfg.runpod.one_shot:
    #     logging.info('trainer.add_callback(OneshotCallback)')
    #     trainer.add_callback(OneshotCallback)

    logging.info('setup_trainer_ex after')

    log_info(f"Training started: {wandb.run.get_url()}")
    
    return trainer

finetune.setup_trainer = setup_trainer_ex

if __name__ == "__main__":
    fire.Fire(train_ex)

    