from math import ceil

import torch
import torch.nn as nn

from src.networks.modules import Bottom, Decoder, Encoder
from src.networks.utils import PositionalEncoder


### UNET 2D
class UNet2D(nn.Module):

    def __init__(
        self,
        channels=[1, 64, 128, 256, 512],
        ratios=[2, 2, 2, 2],
        time_channels=128,
        num_labels=10,
        semantic_channels=16,
        n_groups=1,
        device="cpu",
    ):
        super(UNet2D, self).__init__()
        self.channels = channels

        self.encoder = Encoder(
            channels=channels,
            ratios=ratios,
            time_channels=time_channels,
            n_groups=n_groups,
            target_type="image",
            device=device,
        )

        self.bottom = Bottom(
            n_channels=channels[-1],
            time_channels=time_channels,
            n_groups=n_groups,
            target_type="image",
            device=device,
        )

        self.decoder = Decoder(
            channels=channels,
            ratios=ratios,
            time_channels=time_channels,
            n_groups=n_groups,
            target_type="image",
            device=device,
        )

        self.output_layer = nn.Sequential(
            nn.GroupNorm(num_channels=channels[0], num_groups=n_groups, device=device),
            nn.SiLU(),
            nn.Conv2d(
                in_channels=channels[0],
                out_channels=channels[0],
                kernel_size=3,
                padding="same",
                device=device,
            ),
        )

        self.label_embedding = (
            nn.Embedding(
                num_embeddings=num_labels, embedding_dim=time_channels, device=device
            )
            if num_labels
            else None
        )
        self.time_emmbedding = PositionalEncoder(
            num_channels=time_channels, device=device
        )
        self.semantic_embedding = nn.Linear(
            in_features=semantic_channels, out_features=time_channels, device=device
        )

    def forward(self, x, t, labels=None, z_semantic=None):
        emb = self.time_emmbedding(t)

        if labels is not None:
            labels = self.label_embedding(labels)
            emb = emb + labels

        if z_semantic is not None:
            z_semantic = self.semantic_embedding(z_semantic)
            emb = emb + z_semantic

        x, skip_connections = self.encoder(x, emb)
        x = self.bottom(x, emb)
        x = self.decoder(x, emb, skip_connections)
        return self.output_layer(x)


### Semantic encoder for DiffAE
class SemanticEncoder(nn.Module):

    def __init__(
        self,
        channels=[1, 64, 128, 256, 512],
        ratios=[2, 2, 2, 2],
        out_channels=16,
        resolution=64,
        n_groups=1,
        device="cpu",
    ):
        super(SemanticEncoder, self).__init__()
        self.channels = channels  # used as a param for diffusion handler

        self.encoder = Encoder(
            channels=channels,
            ratios=ratios,
            time_channels=None,
            semantic_channels=0,
            n_groups=n_groups,
            target_type="image",
            device=device,
        )
        embedding_in_channels = (
            int(channels[-1] * ceil(max(resolution / (2 ** (len(channels) - 1)), 1)))
            * 2
        )
        self.out_emb = nn.Sequential(
            nn.Linear(
                in_features=embedding_in_channels, out_features=128, device=device
            ),
            nn.Linear(in_features=128, out_features=out_channels, device=device),
        )

    def forward(self, x):
        x, _ = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        return self.out_emb(x)
