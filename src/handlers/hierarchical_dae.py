import numpy as np
import torch
from tqdm import tqdm

import src.handlers.utils as handler_utils
import src.networks.utils as network_utils
from src.handlers.base_handler import BaseTrainHandler


class HierarchicalDAE(BaseTrainHandler):

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

        # Training process
        for _, data in enumerate(tqdm(self.train_dataloader, leave=False)):
            z_audio = data["z"].to(self.device)[..., : self.length].float()
            z_audio = handler_utils.zero_pad(z_audio, self.length, self.device)

            z_hierarchical = network_utils.warmup_dropout(
                self.semantic_encoder(z_audio), dropout_max_step, epoch
            )
            z_hierarchical = network_utils.regular_dropout(z_hierarchical, p=0.14)

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
            z_audio = data["z"].to(self.device)[..., : self.length].float()
            z_audio = handler_utils.zero_pad(z_audio, self.length, self.device)

            z_hierarchical = self.semantic_encoder(z_audio)
            loss = self.loss_function.compute_loss(
                self.diffusion_unet, z_audio, z_hierarchical=z_hierarchical
            ).mean()

            running_loss.append(loss.item())

        # Update tensorboard
        self._tensorboard_update_scalars(
            epoch=epoch, running_loss=np.mean(running_loss)
        )

        valid_data = next(iter(self.valid_dataloader))
        valid_data = valid_data["z"][..., : self.length][:2].to(self.device).float()
        valid_data = handler_utils.zero_pad(valid_data, self.length, self.device)
        self._tensorboard_update_media(
            epoch=epoch, valid_data=valid_data, sample_rate=self.sample_rate
        )
