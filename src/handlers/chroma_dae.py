import numpy as np
import torch
from tqdm import tqdm

import src.handlers.utils as handler_utils
import src.networks.utils as network_utils
from src.handlers.base_handler import BaseTrainHandler


class ChromaDAE(BaseTrainHandler):

    def __init__(
        self,
        semantic_encoder,
        diffusion_unet,
        emb_model,
        optimizer,
        train_dataloader,
        valid_dataloader,
        loss_function,
        sampler,
        length,
        model_path,
        log_dir,
        data_selector,
        device="cpu",
    ) -> None:
        super().__init__(
            semantic_encoder,
            diffusion_unet,
            emb_model,
            optimizer,
            train_dataloader,
            valid_dataloader,
            loss_function,
            sampler,
            length,
            model_path,
            log_dir,
            data_selector,
            device,
        )

    def _train_one_epoch(
        self, epoch, log_freq, valid_log_freq, save_freq, dropout_max_step, stage
    ):
        running_loss = []
        # Define log states for training and validation
        log = ((epoch) % log_freq == 0) and (epoch != 0)
        valid_log = ((epoch) % valid_log_freq == 0) and (epoch != 0)

        # Authorize next stages to train
        if stage == 1:
            activate_encoder_02 = False
            activate_encoder_03 = False
        if stage == 2:
            for _, p in enumerate(self.semantic_encoder.encoder_01.parameters()):
                p.requires_grad_(False)
            activate_encoder_02 = True
            activate_encoder_03 = False
        if stage == 3:
            for _, p in enumerate(self.semantic_encoder.encoder_01.parameters()):
                p.requires_grad_(False)
            for _, p in enumerate(self.semantic_encoder.encoder_02.parameters()):
                p.requires_grad_(False)
            activate_encoder_02 = True
            activate_encoder_03 = True

        # Training process
        for _, data in enumerate(tqdm(self.train_dataloader, leave=False)):
            waveform = (
                data["waveform"][..., : self.audio_length].to(self.device).float()
            )
            z_audio = data["z"].to(self.device)[..., : self.length].float()
            z_audio = handler_utils.zero_pad(z_audio, self.length, self.device)

            z_hierarchical = network_utils.regular_dropout(
                self.semantic_encoder(
                    waveform, activate_encoder_02, activate_encoder_03
                ),
                p=0.14,
            )
            loss = self.loss_function.compute_loss(
                self.diffusion_unet, z_audio, z_hierarchical=z_hierarchical
            ).mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            running_loss.append(loss.item())

        # Log to tensorboard
        self.log(epoch, log, valid_log, save_freq, running_loss)

    @torch.no_grad()
    def validate(self, epoch):
        running_loss = []
        # Compute and log validation loss
        for _, data in enumerate(self.valid_dataloader):
            waveform = data["waveform"][:, : self.audio_length].to(self.device).float()
            z_audio = data["z"].to(self.device)[..., : self.length].float()
            z_audio = handler_utils.zero_pad(z_audio, self.length, self.device)

            z_hierarchical = self.semantic_encoder(waveform)
            loss = self.loss_function.compute_loss(
                self.diffusion_unet, z_audio, z_hierarchical=z_hierarchical
            ).mean()

            running_loss.append(loss.item())

        # Update tensorboard
        self._tensorboard_update_scalars(
            epoch=epoch, running_loss=np.mean(running_loss)
        )
        valid_data = next(iter(self.valid_dataloader))
        valid_data = (
            valid_data["waveform"][..., : self.audio_length][:2].to(self.device).float()
        )
        self._tensorboard_update_media(
            epoch=epoch, valid_data=valid_data, sample_rate=self.sample_rate
        )

    # Update these functions to this specific case
    @torch.no_grad()
    def encode_decode(self, waveform):
        # Get encoded z_sem representation of x_0
        x_T = network_utils.sample_noise(
            "audio", waveform.size(0), self.num_channels, self.length, self.device
        )
        z_hierarchical = self.semantic_encoder(waveform)
        return self.sampler.sample(
            self.diffusion_unet, x_T, z_hierarchical=z_hierarchical
        )

    def mix_audios(self, x_1, x_2, x_3):
        # Get encoded z_sem representation of x_0
        x_T = network_utils.sample_noise(
            "audio", 2, self.num_channels, self.length, self.device
        )
        z_hierarchical = self.semantic_encoder.encode_inference(x_1, x_2, x_3)
        return self.sampler.sample(
            self.diffusion_unet, x_T, z_hierarchical=z_hierarchical
        )

    def _tensorboard_update_media(self, epoch, valid_data, sample_rate):
        reconstruction, _ = self.encode_decode(valid_data)

        self.logger.add_audio(
            "audio_ground_truth",
            valid_data[0],
            global_step=epoch,
            sample_rate=sample_rate,
        )
        self.logger.add_audio(
            "audio_reconstruction",
            self.emb_model.decode(reconstruction)[0].squeeze(),
            global_step=epoch,
            sample_rate=sample_rate,
        )
        self.logger.flush()
        self.logger.close()
