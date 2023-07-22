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
from accelerate import Accelerator
from accelerate.tracking import on_main_process
import torch
import numpy as np

from huggingface_hub import login
login(os.environ.get("HUGGINGFACE_TOKEN"), add_to_git_credential=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
axolotl_root = os.getenv("AXOLOTL_ROOT", os.path.abspath(os.path.join(project_root, "../axolotl")))
src_dir = os.path.join(axolotl_root, "src")
scripts_dir = os.path.join(axolotl_root, "scripts")
sys.path.insert(0, src_dir)
sys.path.insert(0, scripts_dir)

import finetune
import axolotl
from axolotl.utils.trainer import setup_trainer as setup_trainer_orig
from axolotl.utils.models import load_tokenizer as load_tokenizer_orig, load_model as load_model_orig
from axolotl.utils.dict import DictDefault

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

context = {}

# TODO: avoid code dup
def notify_discord(msg):
    webhook = SyncWebhook.from_url(os.getenv("DISCORD_WEBHOOK_URL"))
    webhook.send(msg)

def edit_discord_message(last_msg, msg):
    return last_msg.edit(content=msg)

def log_info(msg):
    logging.info(msg)
    notify_discord(msg)

def log_error(msg, exc_info=None):
    logging.error(msg, exc_info=exc_info)
    if exc_info is not None:
        notify_discord(f'{msg}: {exc_info}')
    else:
        notify_discord(msg)

def local_rank():
    return int(os.environ.get("LOCAL_RANK", 0))

def parse_config(config, kwargs):
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

    return cfg

def init_accelerator_with_trackers(cfg):
    # os.environ["WANDB_RESUME"] = "auto"
    if cfg.wandb_project is not None:
        run_id = cfg.wandb_run_id or wandb.util.generate_id()
        accelerator = Accelerator(log_with="wandb")
        accelerator.init_trackers(cfg.wandb_project, init_kwargs={"id": run_id})
        # run = wandb.init(project=cfg.wandb_project, id=run_id) #, resume=True)
        os.environ["WANDB_RUN_ID"] = run_id
        return accelerator
    else:
        accelerator = Accelerator()
        return accelerator

def train_ex(
    config,
    prepare_ds_only: bool = False,
    **kwargs,
):
    config = Path(config.strip())

    if local_rank() == 0:
        log_info(f"Prepare training with config: {config}")

    cfg = parse_config(config, kwargs)

    accelerator = init_accelerator_with_trackers(cfg)

    try:
        logging.info('train_ex before')
        finetune.train(config, prepare_ds_only, **kwargs)
        logging.info('train_ex after')

    except Exception as ex:
        log_error(f"Error during training: {ex}", exc_info=ex)
    finally:
        accelerator.end_training()

        # If we need it stay alive for inspection, we should set one_shot to false
        if cfg.runpod.one_shot:        
            runpod.api_key = os.getenv("RUNPOD_API_KEY")

            pod_id = os.getenv("RUNPOD_POD_ID")

            runpod.terminate_pod(pod_id)

            log_info(f"Pod {pod_id} terminated on train end")

def log_data(name, data, tokenizer):
    # logging.info(f'{name}(type={type(data)}, shape={data.shape}):\n{data}')

    try:
        if data.ndim == 3:
            data = torch.argmax(torch.from_numpy(data), dim=-1)

        if data.ndim != 2:
            raise ValueError(f'Invalid data shape: {type(data)} {data.shape}')
        
        data = np.where(data != -100, data, tokenizer.pad_token_id)

        logging.info(f'{name}:\n{tokenizer.batch_decode(data, skip_special_tokens=True)}')
        
        if wandb.run:
            for i in range(len(data)):
                hist = wandb.Histogram(data[i]) #, num_bins=512)
                wandb.log({f"histogram/{name}": hist})

    except Exception as ex:
        logging.error(f'Error logging {name}: {ex}', exc_info=ex)

def decode_data(name, data, tokenizer):
    try:
        if data.ndim == 3:
            data = torch.argmax(torch.from_numpy(data), dim=-1)

        if data.ndim != 2:
            raise ValueError(f'Invalid data shape: {type(data)} {data.shape}')
        
        data = np.where(data != -100, data, tokenizer.pad_token_id)

        return tokenizer.batch_decode(data)

    except Exception as ex:
        logging.error(f'Error decoding {name}, returning empty strings: {ex}', exc_info=ex)
        return ['' for _ in range(len(data))]

def log_eval_prediction_debug(ep, tokenizer):
    log_data('inputs', ep.inputs, tokenizer)
    log_data('predictions', ep.predictions, tokenizer)
    log_data('labels', ep.label_ids, tokenizer)

def log_eval_prediction(ep, tokenizer):
    if wandb.run:
        try:
            data = {
                'input': decode_data('inputs', ep.inputs, tokenizer),
                'prediction': decode_data('predictions', ep.predictions, tokenizer),
                'labels': decode_data('label_ids', ep.label_ids, tokenizer)
            }

            df = pd.DataFrame(data)
            table = wandb.Table(dataframe=df)

            artifact = wandb.Artifact('eval', type="dataset")
            artifact.add(table, 't_eval')
            wandb.run.log_artifact(artifact)

        except Exception as ex:
            logging.error(f'Error logging eval predictions: {ex}', exc_info=ex)

def setup_trainer_ex(cfg, train_dataset, eval_dataset, model, tokenizer):
    # logging.info(f'cfg.runpod.one_shot = {cfg.runpod.one_shot}')

    if os.environ.get('ACCELERATE_USE_DEEPSPEED', 'false') == 'true':
        cfg.deepspeed = os.environ.get('DEEPSPEED_CONFIG_PATH', False)

    logging.info('setup_trainer_ex before')
    trainer = setup_trainer_orig(cfg, train_dataset, eval_dataset, model, tokenizer)
    logging.info('setup_trainer_ex after')

    trainer.args.include_inputs_for_metrics = True
    compute_metrics_orig = trainer.compute_metrics

    tokenizer = context['tokenizer']

    logging.info(f'trainer.tokenizer: {tokenizer}')

    def compute_metrics(ep):
        metrics = compute_metrics_orig(ep) if compute_metrics_orig else {}
        log_eval_prediction(ep, tokenizer)
        return metrics

    if cfg.runpod.log_eval:
        trainer.compute_metrics = compute_metrics

    # Only the main process can get `wandb.run` for multiple-GPU training.
    if wandb.run:
        log_info(f"Training started: {wandb.run.get_url()}")
    
    return trainer

def load_tokenizer_ex(
    tokenizer_config,
    tokenizer_type,
    cfg,
):
    tokenizer = load_tokenizer_orig(tokenizer_config, tokenizer_type, cfg)
    context['tokenizer'] = tokenizer
    return tokenizer

def load_model_ex(
    base_model, base_model_config, model_type, tokenizer, cfg, adapter="lora"
):
    local_model_path = os.path.join('models', f"{'_'.join(base_model.split('/')[-2:])}")
    local_model_config_path = os.path.join('models', f"{'_'.join(base_model_config.split('/')[-2:])}")

    # The model should be pre-downloaded before this training script
    if Path(local_model_path).exists() and Path(local_model_config_path).exists():
        log_info(f'Loading model from local: base_model={local_model_path} base_model_config={local_model_config_path}')
        model = load_model_orig(local_model_path, local_model_config_path, model_type, tokenizer, cfg, adapter)
    else:
        model = load_model_orig(base_model, base_model_config, model_type, tokenizer, cfg, adapter)

    context['model'] = model

    return model

if __name__ == "__main__":
    finetune.setup_trainer = setup_trainer_ex
    finetune.load_tokenizer = load_tokenizer_ex
    finetune.load_model = load_model_ex
    fire.Fire(train_ex)

    