import torch
import torch.nn as nn
import torch.nn.functional as F

import src.networks.utils as network_utils


#############################################
#   UNET Elementary Blocks
#############################################
class ConvBlock(nn.Module):
    '''
    Base convolutional block for UNet, adaptable both for the cases of image and audio.
    Param: target_type : "image" for 2D or "audio" for 1D.
    '''

    def __init__(self, in_channels, skip_channels, out_channels, time_channels,
                 semantic_channels, n_groups, target_type, device):
        super(ConvBlock, self).__init__()
        assert (target_type
                in ["audio",
                    "image"]), "target_type must be either audio or image."
        self.device = device
        self.out_channels = out_channels

        if time_channels is not None:
            self.norm = network_utils.AdaGN(in_channels=out_channels,
                                            time_emb_channels=time_channels,
                                            n_groups=n_groups,
                                            target_type=target_type,
                                            device=self.device)

        if target_type == "image":
            self.conv1 = nn.Conv2d(in_channels=in_channels + skip_channels,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=1,
                                   device=self.device)
            self.conv2 = nn.Conv2d(in_channels=out_channels +
                                   semantic_channels,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=1,
                                   device=self.device)

        if target_type == "audio":
            self.conv1 = nn.Conv1d(in_channels=in_channels + skip_channels,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=1,
                                   device=self.device)
            self.conv2 = nn.Conv1d(in_channels=out_channels +
                                   semantic_channels,
                                   out_channels=out_channels,
                                   kernel_size=3,
                                   padding=1,
                                   device=self.device)

        self.activation = nn.SiLU()

    def forward(self, x, t=None, y=None, z=None, z_temporal=None, skip=None):
        u = torch.cat((x, skip), dim=1) if skip is not None else x
        u = self.conv1(u)
        u = self.norm(u, t, y, z) if t is not None else u
        u = self.activation(u) + x

        h = torch.cat((u, z_temporal), dim=1) if z_temporal is not None else u
        h = self.conv2(h)
        h = self.norm(h, t, y, z) if t is not None else h
        h = self.activation(h) + u
        return h


class DownSample(nn.Module):
    '''
    Unet downsampling block, for both image and audio. 
    Param: target_type : "image" for 2D or "audio" for 1D.
    '''

    def __init__(self, in_channels, out_channels, ratio, target_type,
                 device) -> None:
        super().__init__()
        assert (target_type
                in ["audio",
                    "image"]), "target_type must be either audio or image."
        # for previous models : 3
        kernel = 2 * (ratio - 1) + 1

        if target_type == "image":
            self.conv = nn.Conv2d(in_channels,
                                  out_channels,
                                  kernel, (ratio, 1),
                                  1,
                                  device=device)

        if target_type == "audio":
            self.conv = nn.Conv1d(in_channels,
                                  out_channels,
                                  kernel,
                                  ratio,
                                  1,
                                  device=device)

    def forward(self, x):
        return self.conv(x)


class UpSample(nn.Module):
    '''
    Unet upsampling block, for both image and audio. 
    Param: target_type : "image" for 2D or "audio" for 1D.
    '''

    def __init__(self, in_channels, out_channels, ratio, target_type,
                 device) -> None:
        super().__init__()
        assert (target_type
                in ["audio",
                    "image"]), "target_type must be either audio or image."
        self.target_type = target_type
        self.ratio = ratio

        if target_type == "image":
            self.up_conv = nn.ConvTranspose2d(
                in_channels, out_channels, 4, ratio, 1,
                device=device) if ratio > 1 else nn.Identity()
            self.regular_conv = nn.Conv2d(out_channels,
                                          out_channels,
                                          1,
                                          padding="same",
                                          device=device)

        if target_type == "audio":
            self.regular_conv = nn.Conv1d(in_channels,
                                          out_channels,
                                          1,
                                          padding="same",
                                          device=device)

    def forward(self, x):
        if self.target_type == "image":
            x = self.up_conv(x)

        if self.target_type == "audio":
            x = F.interpolate(x, scale_factor=self.ratio, mode="nearest")

        return self.regular_conv(x)


#############################################
#   Higher level - Encoding and Decoding blocks
#############################################
class Encoder(nn.Module):
    '''
    UNet encoding process. Alternatively processes input data with convolutional and downsampling blocks.
    Param target_type : "image" for 2D or "audio" for 1D.
    Param channels : array of channel dimensions, ex [64, 128, 256, 512]
    Param z_channels : array of channel dimensions for Semantic Encoder, ex [0, 256, 512, 1024]

    Returns: the encoded data and an array with memory of each floor for skip connections.
    '''

    def __init__(self, channels, ratios, time_channels, semantic_channels,
                 n_groups, indexes, target_type, device):
        super(Encoder, self).__init__()
        assert (target_type
                in ["audio",
                    "image"]), "target_type must be either audio or image."
        self.channels = channels
        self.indexes = indexes

        self.encoder_blocks = []
        self.downsampling_blocks = []
        self.time_emb_blocks = []
        for layer_index in range(len(channels) - 1):
            layer_channels = channels[layer_index]
            next_layer_channels = channels[layer_index + 1]
            ratio = ratios[layer_index]

            if layer_index in self.indexes:
                adapted_semantic_channels = semantic_channels
            else:
                adapted_semantic_channels = 0

            conv_block = ConvBlock(
                in_channels=layer_channels,
                skip_channels=0,
                out_channels=layer_channels,
                time_channels=time_channels,
                semantic_channels=adapted_semantic_channels,
                n_groups=n_groups,
                target_type=target_type,
                device=device,
            )
            downsampling_block = DownSample(
                in_channels=layer_channels,
                out_channels=next_layer_channels,
                ratio=ratio,
                target_type=target_type,
                device=device,
            )

            self.encoder_blocks.append(conv_block)
            self.downsampling_blocks.append(downsampling_block)

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)
        self.downsampling_blocks = nn.ModuleList(self.downsampling_blocks)

    def forward(self, x, t=None, y=None, z=None, z_layers=None):
        skip_connections = []
        for layer_index in range(len(self.channels) - 1):
            z_temporal = None
            if layer_index in self.indexes:
                z_temporal = z_layers[
                    layer_index] if z_layers is not None else None
            x = self.encoder_blocks[layer_index](x, t, y, z, z_temporal)
            skip_connections.append(x)
            x = self.downsampling_blocks[layer_index](x)

        return x, skip_connections


class Bottom(nn.Module):
    '''
    Bottom block of the UNet. Simple convolutional block
    Param target_type : "image" for 2D or "audio" for 1D.
    '''

    def __init__(self, n_channels, time_channels, n_groups, target_type,
                 device):
        super(Bottom, self).__init__()
        assert (target_type
                in ["audio",
                    "image"]), "target_type must be either audio or image."
        self.device = device

        self.conv_block = ConvBlock(in_channels=n_channels,
                                    skip_channels=0,
                                    out_channels=n_channels,
                                    time_channels=time_channels,
                                    semantic_channels=0,
                                    n_groups=n_groups,
                                    target_type=target_type,
                                    device=self.device)

    def forward(self, x, t=None, y=None, z=None, z_temporal=None):
        return self.conv_block(x, t, y, z, z_temporal)


class Decoder(nn.Module):
    '''
    UNet decoding process. Alternatively processes input data with upsampling blocks, skip connections
    extracted from encoding blocks and convolutions.
    Param target_type = "image" for 2D or "audio" for 1D.
    Param channels : array of channel dimensions, ex :[64, 128, 256, 512]

    Returns: the decoded data.
    '''

    def __init__(self, channels, ratios, time_channels, semantic_channels,
                 n_groups, indexes, target_type, device):
        super(Decoder, self).__init__()
        assert (target_type
                in ["audio",
                    "image"]), "target_type must be either audio or image."
        self.channels = channels
        self.indexes = indexes

        self.decoder_blocks = []
        self.time_emb_blocks = []
        self.upsampling_blocks = []

        for layer_index in reversed(range(len(channels) - 1)):
            layer_channels = channels[layer_index]
            ratio = ratios[layer_index]

            if layer_index + 1 in indexes:
                adapted_semantic_channels = semantic_channels
            else:
                adapted_semantic_channels = 0

            upsampling_block = UpSample(in_channels=channels[layer_index + 1],
                                        out_channels=layer_channels,
                                        ratio=ratio,
                                        target_type=target_type,
                                        device=device)
            conv_block = ConvBlock(in_channels=layer_channels,
                                   skip_channels=layer_channels,
                                   out_channels=layer_channels,
                                   time_channels=time_channels,
                                   semantic_channels=adapted_semantic_channels,
                                   n_groups=n_groups,
                                   target_type=target_type,
                                   device=device)

            self.decoder_blocks.append(conv_block)
            self.upsampling_blocks.append(upsampling_block)

        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)
        self.upsampling_blocks = nn.ModuleList(self.upsampling_blocks)

    def forward(self, x, t, skip_connections, y, z, z_layers=None):
        for layer_index in range(len(self.channels) - 1):
            skip = skip_connections.pop()
            z_temporal = None
            if len(self.channels) - (layer_index + 1) in self.indexes:
                # Using repeat(1, 1, 2) to match the new size after skip cat.
                z_temporal = z_layers[len(self.channels) -
                                      (layer_index + 1)].repeat(
                                          1, 1,
                                          2) if z_layers is not None else None

            x = self.upsampling_blocks[layer_index](x)
            x = self.decoder_blocks[layer_index](x, t, y, z, z_temporal, skip)
        return x


class HierarchicalEncoder(nn.Module):
    '''
    Unconditional hierarchical encoding process. Works similarly to the unet encoder but with no conditioning.
    Used for the Semantic encoder in Unet.
    Param target_type = "image" for 2D or "audio" for 1D.
    Param channels : array of channel dimensions, ex :[64, 128, 256, 512]

    Returns: the hierarchcally encoded data in an array.
    '''

    def __init__(self, channels, ratios, n_groups, target_type, device):
        super(HierarchicalEncoder, self).__init__()
        assert (target_type
                in ["audio",
                    "image"]), "target_type must be either audio or image."
        self.channels = channels

        self.encoder_blocks = []
        self.downsampling_blocks = []
        self.time_emb_blocks = []
        for layer_index in range(len(channels) - 1):
            layer_channels = channels[layer_index]
            ratio = ratios[layer_index]

            conv_block = ConvBlock(in_channels=layer_channels,
                                   skip_channels=0,
                                   out_channels=layer_channels,
                                   time_channels=None,
                                   semantic_channels=0,
                                   n_groups=n_groups,
                                   target_type=target_type,
                                   device=device)
            downsampling_block = DownSample(in_channels=layer_channels,
                                            out_channels=channels[layer_index +
                                                                  1],
                                            ratio=ratio,
                                            target_type=target_type,
                                            device=device)

            self.encoder_blocks.append(conv_block)
            self.downsampling_blocks.append(downsampling_block)

        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)
        self.downsampling_blocks = nn.ModuleList(self.downsampling_blocks)

    def forward(self, x):
        output = [x]
        for layer_index in range(len(self.channels) - 1):
            x = self.encoder_blocks[layer_index](x)
            x = self.downsampling_blocks[layer_index](x)
            output.append(x)
        return output


class HierarchicalDecoder(nn.Module):
    '''
    Unconditional hierarchical decoding process. Works similarly to the unet encoder but with no conditioning.
    Used for the Semantic encoder in Unet.
    Param target_type = "image" for 2D or "audio" for 1D.
    Param channels : array of channel dimensions, ex :[64, 128, 256, 512]

    Returns: the hierarchcally decoded data in an array.
    '''

    def __init__(self, channels, ratios, n_groups, target_type, use_skip,
                 device):
        super(HierarchicalDecoder, self).__init__()
        assert (target_type
                in ["audio",
                    "image"]), "target_type must be either audio or image."
        self.channels = channels

        self.decoder_blocks = []
        self.time_emb_blocks = []
        self.upsampling_blocks = []

        for layer_index in reversed(range(len(channels) - 1)):
            upsampling_in_channels = channels[layer_index + 1]
            layer_channels = channels[layer_index]
            ratio = ratios[layer_index]

            upsampling_block = UpSample(in_channels=upsampling_in_channels,
                                        out_channels=layer_channels,
                                        ratio=ratio,
                                        target_type=target_type,
                                        device=device)
            conv_block = ConvBlock(
                in_channels=layer_channels,
                skip_channels=layer_channels if use_skip else 0,
                out_channels=layer_channels,
                time_channels=None,
                semantic_channels=0,
                n_groups=n_groups,
                target_type=target_type,
                device=device)

            self.decoder_blocks.append(conv_block)
            self.upsampling_blocks.append(upsampling_block)

        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)
        self.upsampling_blocks = nn.ModuleList(self.upsampling_blocks)

    def forward(self, x, skip_connections=None):
        output = [x]
        for layer_index in range(len(self.channels) - 1):
            skip = skip_connections.pop() if skip_connections else None
            x = self.upsampling_blocks[layer_index](x)
            x = self.decoder_blocks[layer_index](x, skip=skip)
            output.append(x)
        return list(reversed(output))


class HierarchicalUnet(nn.Module):

    def __init__(self,
                 channels,
                 ratios,
                 n_groups=1,
                 use_skip=True,
                 target_type="audio",
                 device='cpu'):
        super(HierarchicalUnet, self).__init__()
        assert (target_type
                in ["audio",
                    "image"]), "target_type must be either audio or image."
        self.channels = channels
        self.ratios = ratios
        self.use_skip = use_skip

        self.encoder = Encoder(
            channels=channels,
            ratios=ratios,
            time_channels=None,
            semantic_channels=0,
            indexes=[],
            n_groups=n_groups,
            target_type=target_type,
            device=device,
        )

        self.bottom = Bottom(
            n_channels=channels[-1],
            n_groups=n_groups,
            time_channels=None,
            target_type=target_type,
            device=device,
        )

        self.decoder = HierarchicalDecoder(
            channels=channels,
            ratios=ratios,
            n_groups=n_groups,
            target_type=target_type,
            use_skip=use_skip,
            device=device,
        )

    def forward(self, x):
        x, skip = self.encoder(x)
        x = self.bottom(x)
        return self.decoder(x, skip) if self.use_skip else self.decoder(x)


class SpectralEncoder(nn.Module):

    def __init__(self, channels, ratios, time_scale, bottleneck_channels,
                 output_channels, transform, n_groups, device):
        super(SpectralEncoder, self).__init__()
        self.device = device
        self.quantizer = transform
        self.encoder = Encoder(
            channels=channels,
            ratios=ratios,
            time_channels=None,
            semantic_channels=0,
            indexes=[],
            n_groups=n_groups,
            target_type="image",
            device=device,
        )
        self.reduce_dimension = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, time_scale)).to(device),
            nn.SiLU(),
        ).to(device)
        self.output_layer = nn.Sequential(
            nn.Conv1d(channels[-1],
                      bottleneck_channels,
                      1,
                      padding="same",
                      device=device), nn.SiLU(),
            nn.Conv1d(bottleneck_channels,
                      output_channels,
                      1,
                      padding="same",
                      device=device), nn.Tanh()).to(device)

    def forward(self, x):
        x = self.quantizer(x).unsqueeze(1)
        x, _ = self.encoder(x)
        x = self.reduce_dimension(x).squeeze()
        x = self.output_layer(x)
        return x
