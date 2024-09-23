import yaml
from music2latent import EncoderDecoder

import src.handlers.utils as handlers_utils
from src.handlers.chroma_dae import ChromaDAE
from src.handlers.hierarchical_dae import HierarchicalDAE
from src.networks.audio.models import (HierarchicalSemanticEncoder,
                                       SpectralSemanticEncoder, UNet1D)
from src.networks.samplers import AlphaSampler
from src.processing import transforms


def load_music2latent_model(
    model_type,
    config_file_name,
    load_epoch,
    dataset,
    device,
):
    assert model_type in ["hdae", "chroma"]
    assert dataset in ["nsynth", "slakh"]
    emb_model = EncoderDecoder(device=device)

    if model_type == "hdae":
        model_handler = load_hdae(
            config_file_name=config_file_name,
            load_epoch=load_epoch,
            dataset=dataset,
            emb_model=emb_model,
            device=device,
        )

    if model_type == "chroma":
        model_handler = load_chroma(
            config_file_name=config_file_name,
            load_epoch=load_epoch,
            dataset=dataset,
            emb_model=emb_model,
            device=device,
        )

    # Training start infos
    print("============================================")
    print("--------------- Model Infos ----------------")
    print(
        f"Diffusion Unet: {handlers_utils.get_number_params(model_handler.diffusion_unet)} M Params"
    )
    print(
        f"Semantic Encoder: {handlers_utils.get_number_params(model_handler.semantic_encoder)} M Params"
    )
    print("============================================")

    return model_handler


def load_hdae(config_file_name, load_epoch, dataset, emb_model, device):
    with open("diffae/src/configs/hierarchical_dae.yml", "r") as config_file:
        config = yaml.safe_load(config_file)

    CONFIG = config[config_file_name]
    MODEL_PATH = f"src/saved_models/{dataset}:{CONFIG['name']}"

    #### MODEL DEF
    semantic_encoder = HierarchicalSemanticEncoder(
        encoder_type=CONFIG["semantic_encoder"]["encoder_type"],
        channels=CONFIG["semantic_encoder"]["channels"],
        ratios=CONFIG["semantic_encoder"]["ratios"],
        bottleneck_channels=CONFIG["semantic_encoder"]["bottleneck_channels"],
        out_channels=CONFIG["semantic_encoder"]["out_channels"],
        unet_use_skip=CONFIG["semantic_encoder"]["unet_use_skip"],
        device=device,
    )
    diffusion_unet = UNet1D(
        channels=CONFIG["unet_1D"]["channels"],
        ratios=CONFIG["unet_1D"]["ratios"],
        time_channels=CONFIG["unet_1D"]["time_channels"],
        temporal_semantic_channels=CONFIG["semantic_encoder"]["out_channels"],
        indexes=CONFIG["semantic_encoder"]["indexes"],
        device=device,
    )
    sampler = AlphaSampler(max_noise_level=80, num_steps=30, alpha=1.1, device=device)

    #### Process Handler
    model_handler = HierarchicalDAE(
        semantic_encoder=semantic_encoder,
        diffusion_unet=diffusion_unet,
        emb_model=emb_model,
        optimizer=None,
        train_dataloader=None,
        valid_dataloader=None,
        loss_function=None,
        sampler=sampler,
        model_path=MODEL_PATH,
        log_dir=None,
        data_selector="music2latent",
        length=32,
        device=device,
    )

    if load_epoch is not None:
        model_handler.load(MODEL_PATH, extension=f"epoch_{load_epoch}")
    else:
        model_handler.load(MODEL_PATH)
    return model_handler


def load_chroma(config_file_name, load_epoch, dataset, emb_model, device):

    with open("src/configs/chroma_dae.yml", "r") as config_file:
        config = yaml.safe_load(config_file)

    CONFIG = config[config_file_name]
    MODEL_PATH = f"src/saved_models/{dataset}:{CONFIG['name']}"

    quantizer_01 = transforms.MelSpectrogram(
        sample_rate=44100, n_mels=32, hop_length=1024, device=device
    )
    quantizer_02 = transforms.MelSpectrogram(
        sample_rate=44100, n_mels=64, hop_length=512, device=device
    )
    quantizer_03 = transforms.MelSpectrogram(
        sample_rate=44100, n_mels=128, hop_length=256, device=device
    )

    semantic_encoder = SpectralSemanticEncoder(
        encoder_01=CONFIG["semantic_encoder"]["encoder_01"],
        encoder_02=CONFIG["semantic_encoder"]["encoder_02"],
        encoder_03=CONFIG["semantic_encoder"]["encoder_03"],
        quantizer_01=quantizer_01,
        quantizer_02=quantizer_02,
        quantizer_03=quantizer_03,
        device=device,
    )
    diffusion_unet = UNet1D(
        channels=CONFIG["unet_1D"]["channels"],
        ratios=CONFIG["unet_1D"]["ratios"],
        time_channels=CONFIG["unet_1D"]["time_channels"],
        temporal_semantic_channels=CONFIG["semantic_encoder"]["out_channels"],
        indexes=CONFIG["semantic_encoder"]["indexes"],
        device=device,
    )
    sampler = AlphaSampler(num_steps=30, alpha=1.1, device=device)

    #### Process Handler
    model_handler = ChromaDAE(
        semantic_encoder=semantic_encoder,
        diffusion_unet=diffusion_unet,
        emb_model=emb_model,
        optimizer=None,
        train_dataloader=None,
        valid_dataloader=None,
        loss_function=None,
        sampler=sampler,
        model_path=MODEL_PATH,
        log_dir=None,
        data_selector="music2latent",
        length=32,
        device=device,
    )
    if load_epoch is not None:
        model_handler.load(MODEL_PATH, extension=f"epoch_{load_epoch}")
    else:
        model_handler.load(MODEL_PATH)
    return model_handler
