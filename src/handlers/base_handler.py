import sys

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

import src.handlers.utils as handler_utils
import src.networks.utils as network_utils


class BaseTrainHandler:

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
        data_selector="music2latent",
        device="cpu",
    ) -> None:
        assert data_selector in ["encodec", "acids_ae", "music2latent"]
        self.semantic_encoder = semantic_encoder
        self.diffusion_unet = diffusion_unet
        self.emb_model = emb_model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.loss_function = loss_function
        self.sampler = sampler
        self.device = device
        self.num_channels = self.diffusion_unet.channels[0]
        self.length = length
        self.logger = SummaryWriter(log_dir)
        self.model_path = model_path
        self.data_selector = data_selector

        if self.data_selector == "acids_ae":
            self.audio_length = 131072
            self.sample_rate = 44100
        if self.data_selector == "encodec":
            self.audio_length = 81920
            self.sample_rate = 24000
        if self.data_selector == "music2latent":
            self.audio_length = 131072
            self.sample_rate = 44100

    def _train_one_epoch(self, *args):
        raise NotImplementedError

    def train(
        self,
        n_epochs=None,
        start_epoch=0,
        log_freq=5,
        valid_log_freq=20,
        save_freq=100,
        dropout_max_step=200,
        stage=1,
    ):
        try:
            print("Training...")
            if n_epochs is not None:
                for epoch in range(start_epoch, n_epochs + start_epoch):
                    self._train_one_epoch(
                        epoch,
                        log_freq,
                        valid_log_freq,
                        save_freq,
                        dropout_max_step,
                        stage,
                    )
                print("Training stopping, saving model")
                self.checkpoint(extension=f"epoch_{epoch}")
                sys.exit()
            else:
                epoch = start_epoch
                while True:
                    self._train_one_epoch(
                        epoch,
                        log_freq,
                        valid_log_freq,
                        save_freq,
                        dropout_max_step,
                        stage,
                    )
                    epoch += 1
        except KeyboardInterrupt:
            print("Training stopping, saving model")
            self.checkpoint(extension=f"epoch_{epoch}")
            sys.exit()

    @torch.no_grad()
    def validate(self, epoch):
        raise NotImplementedError

    @torch.no_grad()
    def encode_decode(self, x_0):
        x_T = network_utils.sample_noise(
            "audio", x_0.size(0), self.num_channels, x_0.size(2), self.device
        )
        z_hierarchical = self.semantic_encoder(x_0)
        return self.sampler.sample(
            self.diffusion_unet, x_T, z_hierarchical=z_hierarchical
        )

    @torch.no_grad()
    def sample(self, batch_size, z_sem, length):
        x_T = network_utils.sample_noise(
            "audio", batch_size, self.num_channels, length, self.device
        )
        return self.sampler.sample(self.diffusion_unet, x_T, z_hierarchical=z_sem)

    def log(self, epoch, log, valid_log, save_freq, running_loss):
        if log:
            print(f"Epoch {epoch}, training loss: {np.mean(running_loss)}")
            self.logger.add_scalar("training loss", np.mean(running_loss), epoch)
            self.logger.flush()
            self.logger.close()

        if valid_log:
            self.validate(epoch)

        if ((epoch) % save_freq == 0) and (epoch != 0):
            self.checkpoint(extension=f"epoch_{epoch}")

    def _tensorboard_update_scalars(self, epoch, running_loss):
        print(f"Epoch {epoch}, validation loss: {np.mean(running_loss)}")
        self.logger.add_scalar("validation loss", np.mean(running_loss), epoch)

    def _tensorboard_update_media(self, epoch, valid_data, sample_rate):
        reconstruction, _ = self.encode_decode(valid_data)

        self.logger.add_audio(
            "audio_ground_truth",
            self.emb_model.decode(valid_data)[0].squeeze(),
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

    def checkpoint(self, extension=None):
        if extension is not None:
            handler_utils.save_checkpoint(
                self.diffusion_unet,
                self.model_path + ".diffusion_unet." + str(extension),
            )
            handler_utils.save_checkpoint(
                self.semantic_encoder,
                self.model_path + ".semantic_encoder." + str(extension),
            )
        else:
            handler_utils.save_checkpoint(
                self.diffusion_unet, self.model_path + ".diffusion_unet"
            )
            handler_utils.save_checkpoint(
                self.semantic_encoder, self.model_path + ".semantic_encoder"
            )

    def load(self, model_path, extension=None):
        print(f"Loading from {model_path}")
        if extension is not None:
            handler_utils.load_checkpoint(
                self.diffusion_unet, model_path + ".diffusion_unet." + str(extension)
            )
            handler_utils.load_checkpoint(
                self.semantic_encoder,
                model_path + ".semantic_encoder." + str(extension),
            )
        else:
            handler_utils.load_checkpoint(
                self.diffusion_unet, model_path + ".diffusion_unet"
            )
            handler_utils.load_checkpoint(
                self.semantic_encoder, model_path + ".semantic_encoder"
            )
