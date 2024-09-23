import nnAudio.features
import torch
import torchaudio.transforms as transforms
from torchaudio.prototype.transforms import ChromaSpectrogram


class SpectralTransform(torch.nn.Module):

    def __init__(self, sample_rate, device) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.device = device


class MelSpectrogram(SpectralTransform):

    def __init__(self, sample_rate, n_mels, hop_length, device) -> None:
        super().__init__(sample_rate, device)
        self.transform = nnAudio.features.mel.MelSpectrogram(
            sr=sample_rate,
            n_fft=hop_length * 4,
            n_mels=n_mels,
            win_length=hop_length * 4,
            hop_length=hop_length,
            center=True,
        ).to(device=device)

    @torch.no_grad()
    def forward(self, x):
        return torch.log1p(self.transform(x))[..., :-1]


class FoldedCQT(SpectralTransform):

    def __init__(
        self, n_folds, sample_rate, hop_length, n_bins, fmin_index, device
    ) -> None:
        super().__init__(sample_rate, device)
        self.n_folds = n_folds
        self.n_octaves = n_bins // 12
        self.transform = nnAudio.features.cqt.CQT1992v2(
            sr=sample_rate,
            hop_length=hop_length,
            fmin=fmin_index * 32.70,
            n_bins=n_bins,
            bins_per_octave=12,
            center=False,
            pad_mode="constant",
            output_format="Magnitude",
        ).to(device=device)

    @torch.no_grad()
    def forward(self, x):
        assert self.n_octaves % self.n_folds == 0
        fold_size = self.n_octaves // self.n_folds
        blocks = ()
        x = self.transform(x)
        for fold_index in range(self.n_folds):
            blocks = blocks + (
                x[
                    :,
                    fold_index * (fold_size * 12) : (fold_index + 1) * (fold_size * 12),
                    :,
                ],
            )

        return torch.mean(torch.stack(blocks), dim=0)


class Chromagram(SpectralTransform):

    def __init__(self, sample_rate, device, n_fft) -> None:
        super().__init__(sample_rate, device)
        self.transform = ChromaSpectrogram(sample_rate=sample_rate, n_fft=n_fft).to(
            device=device
        )

    @torch.no_grad()
    def forward(self, x):
        return self.transform(x).squeeze(1)


class STFT(SpectralTransform):

    def __init__(self, sample_rate, n_fft, hop_length, device) -> None:
        super().__init__(sample_rate, device)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transform = nnAudio.features.STFT(
            n_fft=n_fft,
            hop_length=hop_length,
            sr=sample_rate,
            fmin=40,
            fmax=9000,
            output_format="Magnitude",
        ).to(device=device)

    @torch.no_grad()
    def forward(self, x):
        return torch.log1p(self.transform(x))[..., :-1, :-1]


class TorchSTFT(SpectralTransform):

    def __init__(self, sample_rate, n_fft, hop_length, device) -> None:
        super().__init__(sample_rate, device)
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.transform = transforms.Spectrogram(
            n_fft=n_fft,
            hop_length=hop_length,
            normalized=True,
        ).to(device)

    @torch.no_grad()
    def forward(self, x):
        return torch.log1p(self.transform(x))[..., :-1, :-1]
