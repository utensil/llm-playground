import os
import runpod
from discord import SyncWebhook
import logging

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))

# TODO: avoid code dup
def notify_discord(msg):
    webhook = SyncWebhook.from_url(os.getenv("DISCORD_WEBHOOK_URL"))
    webhook.send(msg)

def log_info(msg):
    logging.info(msg)
    notify_discord(msg)

runpod.api_key = os.getenv("RUNPOD_API_KEY")

pod_id = os.getenv("RUNPOD_POD_ID")

log_info(f"Pod {pod_id} terminated on train end")

runpod.terminate_pod(pod_id)