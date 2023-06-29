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

logging.basicConfig(level=os.getenv("LOG_LEVEL", "DEBUG"))

# os.environ["RUNPOD_DEBUG"] = 'true'

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

def terminate(pod):
    runpod.terminate_pod(pod['id'])
    log_info(f"Pod {pod['id']} terminated")

def monit_runpod(**kwargs):        
    runpod_api_key = os.getenv("RUNPOD_API_KEY")

    if runpod_api_key is None:
        raise ValueError("No RUNPOD_API_KEY environment variable found")
    
    runpod.api_key = runpod_api_key

    try:
    
        myself = runpod.get_myself()['myself']

        logging.info(f"RunPod overview: {myself}")

        if myself is not None:
            msg = ""
            pods = myself['pods']
            if len(pods) > 0:
                msg = f"{len(pods)} pods running, spending ${myself['currentSpendPerHr']} per hour:\n\n"
                msg += "STATUS\tID          \tCPU%\tMEM%\tGPU%\tVRAM%\tUptime\n"

                idle_count = 0

                for pod in pods:
                    id = pod['id']
                    runtime = pod['runtime']
                    if runtime is not None:
                        uptime = runtime['uptimeInSeconds']
                        stat = pod['latestTelemetry']
                        cpu = stat['cpuUtilization']
                        ram = stat['memoryUtilization']
                        gpu = stat['averageGpuMetrics']['percentUtilization']
                        vram = stat['averageGpuMetrics']['memoryUtilization']

                        if gpu > 0.8 and vram > 0.5:
                            status = 'Train'
                        elif cpu > 0.5:
                            status = 'Load'
                        else:
                            status = 'Idle'
                            idle_count += 1
                            
                        msg += f'{status}\t{id}\t{cpu}%\t{ram}%\t{gpu}%\t{vram}%\t{uptime / 60.0:.2f} min\n'
                    else:
                        msg += f'Booting\t{id}\n'

                logging.info(msg)

                if idle_count > 0:
                    log_info(msg)
                else
                    sys.exit(-1)
                
    except Exception as ex:
        log_error(f"Something went wrong with monit_runpod", exc_info=ex)

if __name__ == "__main__":
    fire.Fire(monit_runpod)