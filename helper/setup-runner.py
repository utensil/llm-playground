import importlib
import fire
import yaml
import logging
import os
import sys
from pathlib import Path
from addict import Dict
import json
import time
import runpod
from datetime import datetime, timezone, timedelta
import signal
from tqdm import tqdm
import re
from discord import SyncWebhook
import pexpect

AXOLOTL_RUNPOD_IMAGE = 'winglian/axolotl-runpod:main-py3.9-cu118-2.0.0'
AXOLOTL_RUNPOD_IMAGE_SIZE_IN_GB = 12.5
AXOLOTL_RUNPOD_IMAGE_SIZE = AXOLOTL_RUNPOD_IMAGE_SIZE_IN_GB * 1024 # In MB
BITS_PER_BYTE = 8
COMPRESSION_RATIO = 0.2


DEFAULT_TEMPLATE_ID = '758uq6u5fc'
MAX_BID_PER_GPU = 2.0

POLL_PERIOD = 5 # 5 seconds
MAX_WAIT_TIME = 60 * 10 # 10 minutes

DEFAULT_STOP_AFTER = 60 * 15 # 15 minutes to prevent accidental starting a pod and forgot to stop
DEFAULT_TERMINATE_AFTER = 60 * 60 * 24 # 24 hours to prevent accidental starting a pod and forgot to terminate

class DictDefault(Dict):
    """
    A Dict that returns None instead of returning empty Dict for missing keys.
    Borrowed from https://github.com/utensil/axolotl/blob/local_dataset/src/axolotl/utils/dict.py
    """

    def __missing__(self, key):
        return None

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

os.chdir(project_root)

# src_dir = os.path.join(project_root, "src")
# sys.path.insert(0, src_dir)

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# os.environ["RUNPOD_DEBUG"] = 'true'

def notify_discord(msg):
    webhook = SyncWebhook.from_url(os.getenv("DISCORD_WEBHOOK_URL"))
    return webhook.send(msg, wait=True)

def edit_discord_message(last_msg, msg):
    return last_msg.edit(content=msg)

def log_info(msg):
    logging.info(msg)
    return notify_discord(msg)

def log_error(msg, exc_info=None):
    logging.error(msg, exc_info=exc_info)
    if exc_info is not None:
        return notify_discord(f'{msg}: {exc_info}')
    else:
        return notify_discord(msg)

def as_yaml(data):
    return f'```yaml\n{yaml.dump(data, allow_unicode=True)}\n```'

def terminate(pod):
    runpod.terminate_pod(pod['id'])
    log_info(f"Pod {pod['id']} terminated")

def train_on_runpod(
    config,
    **kwargs,
):

    config = Path(config.strip())
    log_info(f"Setting up RunPod with config: {config}")
    pexpect.run('gh workflow enable monit.yml')

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

        # get the runpod config
        runpod_cfg = cfg.pop('runpod', None)

        if runpod_cfg is None:
            raise ValueError("No pod config found in config file")
        
        runpod_api_key = os.getenv("RUNPOD_API_KEY")

        if runpod_api_key is None:
            raise ValueError("No RUNPOD_API_KEY environment variable found")
        
        runpod.api_key = runpod_api_key

        gpu = runpod_cfg.gpu or "NVIDIA RTX A5000"

        gpu_info = runpod.get_gpu(gpu)

        # TODO: warn if the bid is too high
        bid_per_gpu = min(gpu_info['lowestPrice']['minimumBidPrice'], runpod_cfg.max_bid_per_gpu or MAX_BID_PER_GPU)

        env = runpod_cfg.env or {}
        env['TRAINING_CONFIG'] = str(config)
        env['AXOLOTL_ROOT'] = runpod_cfg.axolotl_root or '/workspace/axolotl'
        env['DISCORD_WEBHOOK_URL'] = os.getenv("DISCORD_WEBHOOK_URL")

        deepspeed = runpod_cfg.deepspeed or cfg.deepspeed or False

        if deepspeed:
            if str(deepspeed).lower() == 'true':
                deepspeed = config.parent.joinpath('./ds_config.json')
            else:
                deepspeed = config.parent.joinpath(deepspeed)

            env['ACCELERATE_USE_DEEPSPEED'] = 'true'
            env['DEEPSPEED_CONFIG_PATH'] = deepspeed
            log_info(f"Deepspeed enabled, using config: {deepspeed}")

        entry = None
        
        if runpod_cfg.entry is not None:
            # TODO: find a better way to escape the entry
            entry = json.dumps(runpod_cfg.entry)[1:-1]

        if runpod_cfg.stop_after == -1:
            stop_after = None
        else:
            stop_after = (datetime.now(timezone.utc) + timedelta(seconds=runpod_cfg.stop_after or DEFAULT_STOP_AFTER)).strftime('"%Y-%m-%dT%H:%M:%SZ"')

        if runpod_cfg.terminate_after == -1:
            terminate_after = None
        else:
            terminate_after = (datetime.now(timezone.utc) + timedelta(seconds=runpod_cfg.terminate_after or DEFAULT_TERMINATE_AFTER)).strftime('"%Y-%m-%dT%H:%M:%SZ"')

        if runpod_cfg.pod_type == 'INTERRUPTABLE':
            pod = runpod.create_spot_pod(f'Training {config}',
                                        AXOLOTL_RUNPOD_IMAGE,
                                        gpu,
                                        cloud_type=runpod_cfg.cloud_type or "SECURE",
                                        bid_per_gpu=bid_per_gpu,
                                        template_id=runpod_cfg.template_id or DEFAULT_TEMPLATE_ID,
                                        container_disk_in_gb=runpod_cfg.container_disk_in_gb or 50,
                                        volume_in_gb=runpod_cfg.volume_in_gb or 200,
                                        gpu_count=runpod_cfg.gpu_count or 1,
                                        min_vcpu_count=runpod_cfg.min_vcpu_count or 8,
                                        min_memory_in_gb=runpod_cfg.min_memory_in_gb or 29,
                                        min_download=runpod_cfg.min_download or 2000,
                                        min_upload=runpod_cfg.min_upload or 1500,
                                        docker_args=entry,
                                        env=env,
                                        stop_after=stop_after,
                                        terminate_after=terminate_after
                                        )
        else:
            pod = runpod.create_pod(f'Training {config}',
                                        AXOLOTL_RUNPOD_IMAGE,
                                        gpu,
                                        cloud_type=runpod_cfg.cloud_type or "SECURE",
                                        template_id=runpod_cfg.template_id or DEFAULT_TEMPLATE_ID,
                                        container_disk_in_gb=runpod_cfg.container_disk_in_gb or 50,
                                        volume_in_gb=runpod_cfg.volume_in_gb or 200,
                                        gpu_count=runpod_cfg.gpu_count or 1,
                                        min_vcpu_count=runpod_cfg.min_vcpu_count or 8,
                                        min_memory_in_gb=runpod_cfg.min_memory_in_gb or 29,
                                        min_download=runpod_cfg.min_download or 2000,
                                        min_upload=runpod_cfg.min_upload or 1500,
                                        docker_args=entry,
                                        env=env,
                                        stop_after=stop_after,
                                        terminate_after=terminate_after
                                        )
        
        
        if pod is None:
            log_error(f"Failed to create pod for {config}")
            return

        def signal_handler(signal, frame):
            logging.info(f"Keyboard interrupt received, terminating pod {pod['id']}")
            terminate(pod)
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        
        msg_created = log_info(f"Created pod {pod['id']}, waiting for it to start...(at most {MAX_WAIT_TIME} seconds)")
        
        username = pod['machine']['podHostId']
        ssh_command = f'ssh {username}@ssh.runpod.io -i ~/.ssh/id_ed25519'
        codespace_ssh_command = f'username={username} scripts/ssh_runpod.sh'
        
        try:        
            # wait for the pod to start
            pod_info = runpod.get_pod(pod['id'])['pod']

            logging.info(f"More about the pod {pod['id']}: {pod_info}")

            eta = AXOLOTL_RUNPOD_IMAGE_SIZE * BITS_PER_BYTE / pod_info['machine']['maxDownloadSpeedMbps'] + AXOLOTL_RUNPOD_IMAGE_SIZE / COMPRESSION_RATIO / pod_info['machine']['diskMBps']

            logging.info(f" - Estimated time to download and extrace the image: {eta} seconds")
            logging.info(f" - While you're waiting, you can check the status of the pod at https://www.runpod.io/console/pods ")
            logging.info(f" - After started, use the following command to ssh into the pod: {ssh_command}")
            logging.info(f"     or the following command in CodeSpace: {codespace_ssh_command}")

            runtime = None
            waited_time = 0
            is_debug = os.getenv("RUNPOD_DEBUG") or ''
            os.environ["RUNPOD_DEBUG"] = ''

            with tqdm(total=eta) as pbar:
                while runtime is None and waited_time < MAX_WAIT_TIME:
                    pod_info = runpod.get_pod(pod['id'])['pod']
                    runtime = pod_info['runtime']
                    time.sleep(POLL_PERIOD)
                    waited_time += POLL_PERIOD
                    pbar.update(POLL_PERIOD)
                    edit_discord_message(msg_created, f"Created pod {pod['id']}, waited for {waited_time}/{eta:.2f} seconds...")

            os.environ["RUNPOD_DEBUG"] = is_debug

            if runtime is None:
                log_error(f"Pod {pod['id']} failed to start in {MAX_WAIT_TIME} seconds: {pod_info}")
                terminate(pod)

            logging.info(f"Pod {pod['id']} started:\n{as_yaml(pod_info)}")
            edit_discord_message(msg_created, f"Pod {pod['id']} started:\n{as_yaml(pod_info)}")

            # myself = runpod.get_myself()

            # log_info(f"RunPod overview:\n{as_yaml(myself)}")

        except Exception as ex:
            log_error(f"Something went wrong with {pod['id']}", exc_info=ex)
            terminate(pod)

if __name__ == "__main__":
    fire.Fire(train_on_runpod)