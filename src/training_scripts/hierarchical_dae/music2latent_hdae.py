### IMOPRTS
import argparse
import os

import diffae.src.handlers.utils as handler_utils
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader

from src.handlers.hierarchical_dae import HierarchicalDAE
from src.networks.audio.models import HierarchicalSemanticEncoder, UNet1D
from src.networks.samplers import AlphaSampler
from src.networks.utils import CustomScaler, EDMLoss
from src.processing.dataloaders import CachedSimpleDataset

"""
RUN AS :
python -m src.training_scripts.hdae.music2latent_hdae --batch_size 32 --dataset slakh --log_freq 1 --valid_freq 5 --save_freq 10
"""

### Static params
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"DEVICE : {DEVICE}")
LR = 5e-5

with open("src/configs/diffusion_autoencoder_hierarchical.yml", "r") as config_file:
    config = yaml.safe_load(config_file)

### Parser setup
parser = argparse.ArgumentParser()
parser.add_argument("--load_model", dest="load_model", type=bool, default=False)
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
parser.add_argument("--load_epoch", dest="load_epoch", type=int, default=None)
parser.add_argument("--start_epoch", dest="start_epoch", type=int, default=0)


### Main process
def main(args):
    LOAD_MODEL = args.load_model
    LOAD_EPOCH = args.load_epoch if args.load_epoch is not None else None
    START_EPOCH = args.start_epoch
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LOG_FREQ = args.log_freq
    VALID_FREQ = args.valid_freq
    SAVE_FREQ = args.save_freq
    DROPOUT_MAX_STEP = args.dropout_max_step
    DATASET = args.dataset
    CONFIG = config[args.config]

    MODEL_PATH = f"diffae/src/saved_models/{CONFIG['name']}"
    LOG_DIR = f"diffae/src/logs/{CONFIG['name']}"

    TRAIN_DATASET_PATH = f"/path_to/{DATASET}/train"
    TEST_DATASET_PATH = f"/path_to/{DATASET}/validation"
    MODEL_PATH = f"src/saved_models/{DATASET}:{CONFIG['name']}"
    LOG_DIR = f"src/logs/{DATASET}:{CONFIG['name']}"

    train_recache_every = 100000
    train_max_samples = 20000

    ### Data loading
    train_dataset = CachedSimpleDataset(
        TRAIN_DATASET_PATH,
        keys=["waveform", "z", "metadata"],
        max_samples=train_max_samples,
        recache_every=train_recache_every,
    )
    test_dataset = CachedSimpleDataset(
        TEST_DATASET_PATH,
        keys=["waveform", "z", "metadata"],
        max_samples=1000,
        recache_every=None,
    )

    dataloader_train = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=True,
        drop_last=True,
    )
    dataloader_test = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=False,
        drop_last=True,
    )

    #### MODEL DEF
    emb_model = EncoderDecoder(device=DEVICE)
    semantic_encoder = HierarchicalSemanticEncoder(
        encoder_type=CONFIG["semantic_encoder"]["encoder_type"],
        channels=CONFIG["semantic_encoder"]["channels"],
        ratios=CONFIG["semantic_encoder"]["ratios"],
        bottleneck_channels=CONFIG["semantic_encoder"]["bottleneck_channels"],
        out_channels=CONFIG["semantic_encoder"]["out_channels"],
        unet_use_skip=CONFIG["semantic_encoder"]["unet_use_skip"],
        device=DEVICE,
    )
    diffusion_unet = UNet1D(
        channels=CONFIG["unet_1D"]["channels"],
        ratios=CONFIG["unet_1D"]["ratios"],
        time_channels=CONFIG["unet_1D"]["time_channels"],
        temporal_semantic_channels=CONFIG["semantic_encoder"]["out_channels"],
        indexes=CONFIG["semantic_encoder"]["indexes"],
        device=DEVICE,
    )
    parameters = list(semantic_encoder.parameters()) + list(diffusion_unet.parameters())
    optimizer = optim.Adam(parameters, lr=LR)
    loss_function = EDMLoss(target_type="audio", device=DEVICE)
    sampler = AlphaSampler(num_steps=30, alpha=1.1, device=DEVICE)
    """
    ## Custom scalers that learn noise instead of original data
    loss_function.scaler = CustomScaler(loss_function.sigma_data, loss_function.device)
    sampler.scaler = CustomScaler(sampler.sigma_data, sampler.device)
    """
    #### Process Handler
    music2latent_HDAEHandler = HierarchicalDAE(
        semantic_encoder=semantic_encoder,
        diffusion_unet=diffusion_unet,
        emb_model=emb_model,
        optimizer=optimizer,
        train_dataloader=dataloader_train,
        valid_dataloader=dataloader_test,
        loss_function=loss_function,
        sampler=sampler,
        model_path=MODEL_PATH,
        log_dir=LOG_DIR,
        data_selector="music2latent",
        length=32,
        device=DEVICE,
    )
    print("Init params...\n")
    if LOAD_MODEL:
        if LOAD_EPOCH is not None:
            music2latent_HDAEHandler.load(MODEL_PATH, extension=f"epoch_{LOAD_EPOCH}")
        else:
            music2latent_HDAEHandler.load(MODEL_PATH)

    # Training start infos
    print("============================================")
    print("--------------- Model Infos ----------------")
    print(
        f"Diffusion Unet: {handler_utils.get_number_params(music2latent_HDAEHandler.diffusion_unet)} M Params"
    )
    print(
        f"Semantic Encoder: {handler_utils.get_number_params(music2latent_HDAEHandler.semantic_encoder)} M Params"
    )
    print("============================================")

    music2latent_HDAEHandler.train(
        n_epochs=EPOCHS,
        start_epoch=START_EPOCH,
        log_freq=LOG_FREQ,
        valid_log_freq=VALID_FREQ,
        save_freq=SAVE_FREQ,
        dropout_max_step=DROPOUT_MAX_STEP,
    )


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
