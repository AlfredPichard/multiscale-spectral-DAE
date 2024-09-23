import torch
import torch.nn as nn
import torch.nn.functional as F
from encodec import EncodecModel

from src.networks.modules import (Bottom, Decoder, DownSample, Encoder,
                                  HierarchicalEncoder, HierarchicalUnet,
                                  SpectralEncoder)
from src.networks.utils import PositionalEncoder, get_unet_index


class UNet1D(nn.Module):

    def __init__(
        self,
        channels,
        ratios,
        time_channels=None,
        metadata_channels=None,
        semantic_channels=None,
        temporal_semantic_channels=None,
        temporal_semantic_ratios=None,
        indexes=[],
        n_groups=1,
        device="cpu",
    ):
        super(UNet1D, self).__init__()
        self.channels = channels
        self.ratios = ratios
        self.semantic_channels = semantic_channels
        self.temporal_semantic_channels = temporal_semantic_channels
        self.temporal_semantic_ratios = temporal_semantic_ratios

        self.encoder = Encoder(
            channels=channels,
            ratios=ratios,
            time_channels=time_channels,
            semantic_channels=temporal_semantic_channels or 0,
            n_groups=n_groups,
            indexes=indexes,
            target_type="audio",
            device=device,
        )

        self.bottom = Bottom(
            n_channels=channels[-1],
            time_channels=time_channels,
            n_groups=n_groups,
            target_type="audio",
            device=device,
        )

        self.decoder = Decoder(
            channels=channels,
            ratios=ratios,
            time_channels=time_channels,
            semantic_channels=temporal_semantic_channels or 0,
            n_groups=n_groups,
            indexes=indexes,
            target_type="audio",
            device=device,
        )

        self.output_layer = nn.Sequential(
            nn.GroupNorm(num_channels=channels[0], num_groups=n_groups, device=device),
            nn.SiLU(),
            nn.Conv1d(
                in_channels=channels[0],
                out_channels=channels[0],
                kernel_size=3,
                padding="same",
                device=device,
            ),
        )

        self.time_emmbedding = (
            PositionalEncoder(num_channels=time_channels, device=device)
            if time_channels
            else None
        )
        self.label_embedding = (
            nn.Linear(
                in_features=metadata_channels, out_features=time_channels, device=device
            )
            if metadata_channels
            else None
        )
        self.semantic_embedding = (
            nn.Linear(
                in_features=semantic_channels, out_features=time_channels, device=device
            )
            if semantic_channels
            else None
        )

        # Resizing functions that are used to inject encoded data everywhere in the unet
        if self.temporal_semantic_ratios is not None:
            self.max_index = get_unet_index(self.temporal_semantic_ratios, self.ratios)
            self.downsampling_blocks = nn.ModuleList(
                [
                    DownSample(
                        self.temporal_semantic_channels,
                        self.temporal_semantic_channels,
                        self.ratios[k],
                        target_type="audio",
                        device=device,
                    )
                    for k in range(self.max_index, len(self.ratios))
                ]
            )

    def forward(self, x, t=None, y=None, z=None, z_temporal=None, z_hierarchical=None):
        z_layers = None

        if t is not None:
            t = self.time_emmbedding(t)

        if y is not None:
            y = self.label_embedding(y)

        if z is not None:
            z = self.semantic_embedding(z)

        if z_temporal is not None:
            z_layers = []
            current_ratio = 1
            for index, ratio in enumerate(self.ratios):
                if index < self.max_index:
                    repeats = (x.size(-1) // current_ratio) // z_temporal.shape[-1]
                    z_layers.append(
                        torch.repeat_interleave(z_temporal, repeats, dim=-1)
                    )

                elif index == self.max_index:
                    z_layers.append(z_temporal)

                else:
                    z_temporal = self.downsampling_blocks[index - (self.max_index + 1)](
                        z_temporal
                    )
                    z_layers.append(z_temporal)
                current_ratio *= ratio

        if z_hierarchical is not None:
            z_layers = z_hierarchical

        x, skip_connections = self.encoder(x, t, y, z, z_layers)
        x = self.bottom(x, t, y, z)
        x = self.decoder(x, t, skip_connections, y, z, z_layers)
        return self.output_layer(x)


### Semantic encoder for DiffAE
class SemanticEncoder(nn.Module):

    def __init__(
        self,
        channels,
        ratios,
        out_channels,
        n_groups=1,
        encoder_type="basic",
        device="cpu",
    ):
        super(SemanticEncoder, self).__init__()
        assert encoder_type in [
            "basic",
            "temporal",
        ], "encoder_type must be either basic or temporal."
        self.encoder_type = encoder_type
        self.channels = channels  # used as a param for diffusion handler

        self.encoder = Encoder(
            channels=channels,
            ratios=ratios,
            time_channels=None,
            semantic_channels=0,
            n_groups=n_groups,
            target_type="audio",
            device=device,
        )

        if self.encoder_type == "basic":
            self.out_emb = nn.Sequential(
                nn.Linear(in_features=channels[-1], out_features=128, device=device),
                nn.Linear(in_features=128, out_features=out_channels, device=device),
            )

        if self.encoder_type == "temporal":
            self.out_emb = nn.Conv1d(
                channels[-1], out_channels, 1, padding="same", device=device
            )

    def forward(self, x):
        x, _ = self.encoder(x)
        if self.encoder_type == "basic":
            x = torch.mean(x, dim=-1)
        return torch.tanh(self.out_emb(x))


class HierarchicalSemanticEncoder(nn.Module):

    def __init__(
        self,
        channels,
        ratios,
        bottleneck_channels,
        out_channels,
        n_groups=1,
        encoder_type="E",
        unet_use_skip=True,
        device="cpu",
    ):
        super(HierarchicalSemanticEncoder, self).__init__()
        assert encoder_type in [
            "E",
            "U",
        ], "Encoder type must be either E (simple encoder) or U (unet encoder)"
        self.encoder_type = encoder_type
        self.channels = channels
        self.ratios = ratios

        if self.encoder_type == "E":
            self.get_latent_representation = HierarchicalEncoder(
                channels=channels,
                ratios=ratios,
                n_groups=n_groups,
                target_type="audio",
                device=device,
            )
        elif self.encoder_type == "U":
            self.get_latent_representation = HierarchicalUnet(
                channels=channels,
                ratios=ratios,
                use_skip=unet_use_skip,
                n_groups=n_groups,
                target_type="audio",
                device=device,
            )

        self.output_embeddings = nn.ModuleList(
            [
                nn.Sequential(
                    nn.GroupNorm(
                        num_channels=channels[k], num_groups=n_groups, device=device
                    ),
                    nn.SiLU(),
                    nn.Conv1d(
                        channels[k],
                        bottleneck_channels[k],
                        1,
                        padding="same",
                        device=device,
                    ),
                    nn.Tanh(),
                    nn.Conv1d(
                        bottleneck_channels[k],
                        out_channels,
                        1,
                        padding="same",
                        device=device,
                    ),
                )
                for k in range(len(self.channels))
            ]
        )

    def forward(self, x, return_embedding=False):
        hierarchical_output = self.get_latent_representation(x)

        out, full_out = [], []
        for k, net in enumerate(self.output_embeddings):
            curz = hierarchical_output[k]
            for layer in net[:-1]:
                curz = layer(curz)

            curz_final = net[-1](curz)

            out.append(curz)
            full_out.append(curz_final)

        if return_embedding:
            return out, full_out
        else:
            return full_out


class SpectralSemanticEncoder(nn.Module):

    def __init__(
        self,
        encoder_01,
        quantizer_01,
        encoder_02,
        quantizer_02,
        encoder_03,
        quantizer_03,
        device,
    ) -> None:
        super().__init__()
        self.encoder_01 = SpectralEncoder(
            channels=encoder_01["channels"],
            ratios=encoder_01["ratios"],
            time_scale=encoder_01["time_scale"],
            bottleneck_channels=encoder_01["bottleneck_channels"],
            output_channels=encoder_01["output_channels"],
            transform=quantizer_01,
            n_groups=1,
            device=device,
        )
        self.encoder_02 = SpectralEncoder(
            channels=encoder_02["channels"],
            ratios=encoder_02["ratios"],
            time_scale=encoder_02["time_scale"],
            bottleneck_channels=encoder_02["bottleneck_channels"],
            output_channels=encoder_02["output_channels"],
            transform=quantizer_02,
            n_groups=1,
            device=device,
        )
        self.encoder_03 = SpectralEncoder(
            channels=encoder_03["channels"],
            ratios=encoder_03["ratios"],
            time_scale=encoder_03["time_scale"],
            bottleneck_channels=encoder_03["bottleneck_channels"],
            output_channels=encoder_03["output_channels"],
            transform=quantizer_03,
            n_groups=1,
            device=device,
        )
        self.upsample = nn.Upsample(scale_factor=2, mode="linear")

    def dropout_factor(self, activate):
        return 0 if not activate else 1

    def forward(self, x, activate_encoder_02=True, activate_encoder_03=True):
        z_1 = self.encoder_01(x)
        z_2 = self.encoder_02(x) * self.dropout_factor(
            activate_encoder_02
        ) + self.upsample(z_1)
        z_3 = self.encoder_03(x) * self.dropout_factor(
            activate_encoder_03
        ) + self.upsample(z_2)

        return [None, z_3, z_2, z_1]

    def encode_inference(self, x_1, x_2, x_3):
        z_1 = self.encoder_01(x_1).unsqueeze(0)
        z_2 = self.encoder_02(x_2).unsqueeze(0) + self.upsample(z_1)
        z_3 = self.encoder_03(x_3).unsqueeze(0) + self.upsample(z_2)

        return [None, z_3.repeat(2, 1, 1), z_2.repeat(2, 1, 1), z_1.repeat(2, 1, 1)]


### Encodec Checkpoint
class WrappedEncodec(nn.Module):

    def __init__(self, params_file=None) -> None:
        super().__init__()
        self.model = EncodecModel.encodec_model_24khz()
        self.model.set_target_bandwidth(24)
        self.sr = 24000
        self.param = False
        if params_file is not None:
            self.param = True
            pca_params = torch.load(params_file)
            self.latent_mean = pca_params["mean"]
            self.components = pca_params["components"]
            self.n_latents = 64

    def truncate(self, z: torch.Tensor):
        z = z - self.latent_mean.unsqueeze(-1)
        z = F.conv1d(z, self.components.unsqueeze(-1).to(z))
        z = z[:, : self.n_latents]
        return z

    def encode(self, x: torch.Tensor, trunc=True) -> torch.Tensor:
        z, scales = self.model._encode_frame(x)
        z = z.transpose(0, 1)  # (n_quant, batch, time)
        z = self.model.quantizer.decode(z)
        if trunc == True and self.param:
            z = self.truncate(z)

        return z

    def decode(self, z: torch.Tensor, trunc=True) -> torch.Tensor:
        if trunc == True and self.param:
            noise = torch.zeros(z.shape[0], 128 - self.n_latents, z.shape[-1]).to(z)
            z = torch.cat((z, noise), axis=1)
            z = F.conv1d(
                z, self.components.T.unsqueeze(-1).to(z)
            ) + self.latent_mean.unsqueeze(-1).to(z)

        return self.model.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))
