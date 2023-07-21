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

from huggingface_hub import login
login(os.environ.get("HUGGINGFACE_TOKEN"), add_to_git_credential=True)

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
axolotl_root = os.getenv("AXOLOTL_ROOT", os.path.abspath(os.path.join(project_root, "../axolotl")))
src_dir = os.path.join(axolotl_root, "src")
scripts_dir = os.path.join(axolotl_root, "scripts")
sys.path.insert(0, src_dir)
sys.path.insert(0, scripts_dir)

import finetune
from axolotl.utils.trainer import setup_trainer as setup_trainer_orig
from axolotl.utils.dict import DictDefault

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

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

    # the following is intensionally not in the finally block, because we want the pod to stay alive for inspection and debugging if anything goes wrong
    if cfg.runpod.one_shot:        
        runpod.api_key = os.getenv("RUNPOD_API_KEY")

        pod_id = os.getenv("RUNPOD_POD_ID")

        runpod.terminate_pod(pod_id)

        log_info(f"Pod {pod_id} terminated on train end")

def log_data(name, data):
    logging.info(f'{name}(type={type(data)}, shape={data.shape}):\n{data}')
    for i in range(len(data)):
        hist = wandb.Histogram(data[i]) #, num_bins=512)
        wandb.log({f"histogram/{name}": hist})

def log_eval_prediction(ep):
    data = {
        'input': ep.inputs,
        'prediction': ep.predictions,
        'label_id': ep.label_ids
    }

    log_data('inputs', ep.inputs)
    log_data('predictions', ep.predictions)
    log_data('label_ids', ep.label_ids)

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
    # logging.info(f'cfg.runpod.one_shot = {cfg.runpod.one_shot}')

    if os.environ.get('ACCELERATE_USE_DEEPSPEED', 'false') == 'true':
        cfg.deepspeed = os.environ.get('DEEPSPEED_CONFIG_PATH', False)

    logging.info('setup_trainer_ex before')
    trainer = setup_trainer_orig(cfg, train_dataset, eval_dataset, model, tokenizer)
    logging.info('setup_trainer_ex after')

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

    # Only the main process can get `wandb.run` for multiple-GPU training.
    if wandb.run:
        log_info(f"Training started: {wandb.run.get_url()}")
    
    return trainer

if __name__ == "__main__":
    finetune.setup_trainer = setup_trainer_ex
    fire.Fire(train_ex)

    