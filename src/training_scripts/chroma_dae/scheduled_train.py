# Call this file pycron.py.
import argparse
import os
import os.path
import time

import diffae.src.training_scripts.chroma_diffae.music2latent_chroma_dae as worker
import schedule
import torch
import yaml

"""
python -m diffae.src.training_scripts.chroma_diffae.scheduled_train --batch_size 8 --log_freq 1 --valid_freq 2 --save_freq 5 --stage 2 --config music2latent_chroma_hdae_02 --dataset slakh --load_model True --load_epoch 50 --start_epoch 51
"""

MODEL_TYPE = "chroma_dae"
LR = 5e-5

with open(f"src/configs/{MODEL_TYPE}.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

### Parser setup
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", dest="batch_size", type=int, default=128)
parser.add_argument("--epochs", dest="epochs", type=int, default=None)
parser.add_argument("--log_freq", dest="log_freq", type=int, default=5)
parser.add_argument("--valid_freq", dest="valid_freq", type=int, default=20)
parser.add_argument("--save_freq", dest="save_freq", type=int, default=20)
parser.add_argument("--dropout_max_step", dest="dropout_max_step", type=int, default=0)
parser.add_argument(
    "--config", dest="config", type=str, default="music2latent_mel_hdae_01"
)
parser.add_argument("--dataset", dest="dataset", type=str, default="slakh")
parser.add_argument("--stage", dest="stage", type=int, default=1)
parser.add_argument("--load_model", dest="load_model", type=bool, default=False)
parser.add_argument("--load_epoch", dest="load_epoch", type=int, default=None)
parser.add_argument("--start_epoch", dest="start_epoch", type=int, default=0)


def get_time() -> str:
    r"""Get current time and date and return as string."""
    return time.strftime("%X (%d/%m/%y)")


def trigger_train_job(args):
    TIME = get_time()
    CONFIG = config[config]
    LOAD_EPOCH = args.load_epoch if args.load_epoch is not None else None
    DATASET = args.dataset
    MODEL_PATH = f"src/saved_models/{DATASET}:{CONFIG['name']}"

    print(f"{TIME}:looking for existing checkpoint...")
    semantic_encoder_path = MODEL_PATH + f".semantic_encoder.epoch_{LOAD_EPOCH}.pt"
    diffusion_unet_path = MODEL_PATH + f".diffusion_unet.epoch_{LOAD_EPOCH}.pt"
    check_file = os.path.isfile(semantic_encoder_path) and os.path.isfile(
        diffusion_unet_path
    )

    if check_file:
        print("Found checkpoint for model - lauching next stage training")
        worker.run(args=args)
        time.sleep(36000)  # Sleep 10 hours.

    else:
        print("No checkpoint found, next check in 1 hour...")


if __name__ == "__main__":
    args = parser.parse_args()
    schedule.every(1).hour.do(lambda: trigger_train_job(args))

    RUN = True
    while RUN:
        try:
            schedule.run_pending()

        except KeyboardInterrupt:
            RUN = False
    print("EXIT")
