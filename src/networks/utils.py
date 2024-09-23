import numpy as np
import torch
import torch.nn as nn
from scipy import signal

#############################################
"""
Utils functions
"""


#############################################
def to_adapted_float_tensor(x, size, target_type, device):
    if target_type == "audio":
        return (x * torch.ones((size, 1, 1), device=device)).float()

    if target_type == "image":
        return (x * torch.ones((size, 1, 1, 1), device=device)).float()


def sample_noise(target_type, batch_size, num_channels, res, device):
    # Sample random noise x_T
    if target_type == "audio":
        return torch.randn(batch_size, num_channels, res, device=device)

    if target_type == "image":
        return torch.randn(batch_size, num_channels, res, res, device=device)

    return None


def get_unet_index(encoder_ratios, unet_ratios):
    encoder_ratio = np.prod(encoder_ratios)
    current_ratio = 1
    for index in range(len(unet_ratios)):
        if current_ratio == encoder_ratio:
            return index
        current_ratio *= unet_ratios[index]

    return len(unet_ratios)


def warmup_dropout(tensor_list, max_epoch, epoch):
    insert_step = max_epoch // len(tensor_list)
    output = [tensor_list[0]]
    for k in range(1, len(tensor_list)):
        current_tensor = tensor_list[k]
        output.append(
            current_tensor if epoch > k * insert_step else torch.zero_(current_tensor)
        )
    return output


def regular_dropout(tensor_list, p):
    for tensor in tensor_list:
        rdn_pool = np.random.random()
        if rdn_pool < p:
            tensor = -2.0 * torch.ones_like(tensor) if tensor is not None else None
    return tensor_list


def phasor(timesteps, encodec_sample_rate=24000, frame_sample_rate=320, n_frames=1024):
    resampled_timesteps = np.floor(
        np.array(timesteps) * encodec_sample_rate / frame_sample_rate
    ).astype(int)
    sawtooth_like = np.zeros(n_frames)

    for k in range(1, len(resampled_timesteps)):
        try:
            t = np.linspace(0, 1, (resampled_timesteps[k] - resampled_timesteps[k - 1]))
            sawtooth_like[resampled_timesteps[k - 1] : resampled_timesteps[k]] = (
                0.5 * signal.sawtooth(2 * np.pi * t) + 0.5
            )
        except ValueError:
            pass
    try:
        last_t = np.linspace(
            0,
            1,
            max(
                0,
                min(
                    resampled_timesteps[-1] - resampled_timesteps[-2],
                    n_frames - resampled_timesteps[-1],
                ),
            ),
        )
        padding = np.zeros(max(0, n_frames - resampled_timesteps[-1] - len(last_t)))
        sawtooth_like[resampled_timesteps[-1] : n_frames] = np.concatenate(
            (0.5 * signal.sawtooth(2 * np.pi * last_t) + 0.5, padding)
        )
    except IndexError:
        pass

    return sawtooth_like


def phasor_from_bpm(
    bpm, encodec_sample_rate=24000, frame_sample_rate=320, n_frames=1024
):
    total_length_seconds = n_frames * frame_sample_rate / encodec_sample_rate
    one_timestep = 60 / bpm
    n_timesteps = np.floor(total_length_seconds / one_timestep).astype(int)
    timesteps = [one_timestep * i for i in range(n_timesteps)]
    return phasor(timesteps, encodec_sample_rate, frame_sample_rate, n_frames)


def local_freeze(model):
    for param in model.parameters():
        param.requires_grad = False


#############################################
"""
Low level - Time Embedding and Group Normalization
"""


#############################################
class PositionalEncoder(torch.nn.Module):

    def __init__(self, num_channels, max_length=10000, device="cpu"):
        super().__init__()
        self.num_channels = num_channels
        self.max_length = max_length
        self.device = device

    def forward(self, x):
        x = x.squeeze() * 500
        omega = torch.arange(
            start=0, end=self.num_channels // 2, dtype=torch.float32, device=self.device
        )
        omega = omega / (self.num_channels // 2)
        omega = (1 / self.max_length) ** omega

        x = x.outer(omega.to(x.dtype))
        x = torch.cat([x.cos(), x.sin()], dim=1)
        return x


class AdaGN(nn.Module):

    def __init__(
        self, in_channels, time_emb_channels, target_type, n_groups=None, device="cpu"
    ):
        super(AdaGN, self).__init__()
        assert target_type in [
            "audio",
            "image",
        ], "target_type must be either audio or image."
        self.target_type = target_type
        if n_groups is None:
            n_groups = in_channels
        self.device = device

        self.groupnorm = nn.GroupNorm(
            n_groups, in_channels, affine=True, eps=1e-5, device=self.device
        )
        self.time_embedding = nn.Linear(
            time_emb_channels, 2 * in_channels, device=self.device
        )
        self.label_embedding = nn.Linear(
            time_emb_channels, in_channels, device=self.device
        )
        self.semantic_embedding = nn.Linear(
            time_emb_channels, in_channels, device=self.device
        )
        self.in_channels = in_channels

    def forward(self, x, t, y=None, z=None):
        if y is not None:
            y = (
                self.label_embedding(y)[:, :, None, None]
                if self.target_type == "image"
                else self.label_embedding(y)[:, :, None]
            )
        else:
            y = 1
        if z is not None:
            z = (
                self.semantic_embedding(z)[:, :, None, None]
                if self.target_type == "image"
                else self.semantic_embedding(z)[:, :, None]
            )
        else:
            z = 1

        t = (
            self.time_embedding(t)[:, :, None, None]
            if self.target_type == "image"
            else self.time_embedding(t)[:, :, None]
        )
        t_mult, t_add = torch.split(t, split_size_or_sections=self.in_channels, dim=1)

        return (t_mult * self.groupnorm(x) + t_add) * y * z


#############################################
"""
Scaler functions
"""


#############################################
class EDMScaler:

    def __init__(self, sigma_data=0.5, device="cpu") -> None:
        self.sigma_data = sigma_data
        self.device = device

    def eval(
        self,
        model,
        batch_size,
        x,
        sigma,
        y=None,
        z=None,
        z_temporal=None,
        z_hierarchical=None,
    ):
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2).sqrt()
        c_noise = sigma.squeeze().log() / 4
        c_noise = c_noise * torch.ones(batch_size, device=self.device)
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)

        return c_skip * x + c_out * model(
            c_in * x, c_noise, y, z, z_temporal, z_hierarchical
        )


class CustomScaler:

    def __init__(self, sigma_data=0.5, device="cpu") -> None:
        self.sigma_data = sigma_data
        self.device = device

    def eval(
        self,
        model,
        batch_size,
        x,
        sigma,
        y=None,
        z=None,
        z_temporal=None,
        z_hierarchical=None,
    ):
        c_in = 1 / (self.sigma_data**2 + sigma**2).sqrt()
        c_noise = sigma.squeeze().log() / 4
        c_noise = c_noise * torch.ones(batch_size, device=self.device)

        return model(c_in * x, c_noise, y, z, z_temporal, z_hierarchical)


#############################################
"""
Loss functions
"""


#############################################
class EDMLoss:

    def __init__(
        self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, target_type="image", device="cpu"
    ) -> None:
        assert target_type in [
            "audio",
            "image",
        ], "target_type must be either audio or image."
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        self.target_type = target_type
        self.device = device

        self.scaler = EDMScaler(self.sigma_data, self.device)

    def compute_loss(
        self,
        model,
        x_0,
        labels=None,
        z_semantic=None,
        z_temporal=None,
        z_hierarchical=None,
    ):
        batch_size = x_0.shape[0]
        if self.target_type == "audio":
            noise = torch.randn([batch_size, 1, 1]).to(self.device)
        if self.target_type == "image":
            noise = torch.randn([batch_size, 1, 1, 1]).to(self.device)

        sigma = (
            (noise * self.P_std + self.P_mean).exp().to(self.device)
        )  # Noise distribution
        loss_weight = (sigma**2 + self.sigma_data**2) / (
            (sigma * self.sigma_data) ** 2
        ).to(self.device)

        n = (torch.randn_like(x_0) * sigma).to(self.device)
        output = self.scaler.eval(
            model,
            batch_size,
            x_0 + n,
            sigma,
            labels,
            z_semantic,
            z_temporal,
            z_hierarchical,
        )
        loss = loss_weight * ((output - x_0) ** 2)
        return loss
