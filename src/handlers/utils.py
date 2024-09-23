import sys

import torch
from torch.utils.tensorboard import SummaryWriter

import src.networks.utils as network_utils


def save_checkpoint(model, model_path):
    model_path = model_path + ".pt"
    # model_path = os.path.join(dir_path, model_path)
    torch.save(obj=model.state_dict(), f=model_path)


def load_checkpoint(model, model_path):
    model_path = model_path + ".pt"
    model.load_state_dict(torch.load(model_path))


def get_number_params(model):
    return sum(p.numel() for p in model.parameters()) / 1e6


def zero_pad(z, length, device):
    if z.shape[2] < length:
        z = torch.cat(
            (z, torch.zeros(z.shape[0], z.shape[1], length - z.shape[2]).to(device)),
            dim=2,
        )[..., :length].float()
    return z


class BaseHandler:

    def __init__(
        self,
        diffusion_unet,
        emb_model,
        optimizer,
        train_dataloader,
        valid_dataloader,
        loss_function,
        sampler,
        model_path,
        log_dir,
        device="cpu",
    ) -> None:
        self.diffusion_unet = diffusion_unet
        self.emb_model = emb_model
        self.optimizer = optimizer
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.loss_function = loss_function
        self.sampler = sampler
        self.device = device
        self.num_channels = self.diffusion_unet.channels[0]
        self.logger = SummaryWriter(log_dir)
        self.model_path = model_path

    def _train_one_epoche(self, *args):
        pass

    @torch.no_grad()
    def encode_decode(self, x_0):
        # Get encoded z_sem representation of x_0
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

    def train(
        self,
        n_epochs=None,
        log_freq=5,
        valid_log_freq=20,
        save_freq=100,
        dropout_max_step=200,
        stage=1,
    ):
        try:
            print("Training...")
            if n_epochs is not None:
                for epoch in range(n_epochs):
                    self._train_one_epoch(
                        epoch,
                        log_freq,
                        valid_log_freq,
                        save_freq,
                        dropout_max_step,
                        stage,
                    )
                print("Training stopping, saving model")
                self.checkpoint()
                sys.exit()
            else:
                epoch = 0
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
            self.checkpoint()
            sys.exit()

    def checkpoint(self):
        save_checkpoint(self.diffusion_unet, self.model_path + ".diffusion_unet")
        save_checkpoint(self.semantic_encoder, self.model_path + ".semantic_encoder")

    def load(self, model_path):
        print(f"Loading from {model_path}")
        load_checkpoint(self.diffusion_unet, model_path + ".diffusion_unet")
        load_checkpoint(self.semantic_encoder, model_path + ".semantic_encoder")
