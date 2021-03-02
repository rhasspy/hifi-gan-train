"""Classes and methods for loading phonemes and mel spectrograms"""
import logging
import random
import typing
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.utils.data
from librosa.filters import mel as librosa_mel_fn
from librosa.util import normalize

from .config import TrainingConfig

_LOGGER = logging.getLogger("hifi_gan_train.dataset")

MAX_WAV_VALUE = 32768.0

# -----------------------------------------------------------------------------

MelKey = typing.Tuple[typing.Optional[float], str]
MelFunction = typing.Callable[[torch.Tensor, typing.Optional[float]], torch.Tensor]


class MelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        wav_paths: typing.List[Path],
        config: TrainingConfig,
        split: bool = True,
        shuffle: bool = True,
    ):
        self.wav_paths = wav_paths

        random.seed(config.seed)
        if shuffle:
            random.shuffle(self.wav_paths)

        self.config = config
        self.segment_size = config.audio.segment_size
        self.sampling_rate = config.audio.sampling_rate
        self.n_fft = config.audio.n_fft
        self.hop_length = config.audio.hop_length
        self.win_length = config.audio.win_length

        self.split = split

        self.mel_bases: typing.Dict[MelKey, typing.Any] = {}
        self.hann_windows: typing.Dict[MelKey, typing.Any] = {}

    def __getitem__(self, index):
        wav_path = self.wav_paths[index]
        audio, _ = librosa.load(wav_path, sr=self.sampling_rate)
        audio = audio / MAX_WAV_VALUE
        audio = normalize(audio) * 0.95

        audio = torch.FloatTensor(audio)
        audio = audio.unsqueeze(0)

        if self.split:
            if audio.size(1) >= self.segment_size:
                max_audio_start = audio.size(1) - self.segment_size
                audio_start = random.randint(0, max_audio_start)
                audio = audio[:, audio_start : audio_start + self.segment_size]
            else:
                audio = torch.nn.functional.pad(
                    audio, (0, self.segment_size - audio.size(1)), "constant"
                )

        mel = self.mel_spectrogram(audio, mel_fmax=self.config.audio.mel_fmax)
        mel_loss = self.mel_spectrogram(audio, mel_fmax=self.config.audio.mel_fmax_loss)

        return (mel.squeeze(), audio.squeeze(0), str(wav_path), mel_loss.squeeze())

    def __len__(self):
        return len(self.wav_paths)

    def mel_spectrogram(
        self, y: torch.Tensor, mel_fmax: typing.Optional[float], center: bool = False
    ) -> torch.Tensor:
        # Look up mel basis and hann window by (fmax, device) key
        key = (mel_fmax, str(y.device))

        mel_basis = self.mel_bases.get(key)
        if mel_basis is None:
            mel_basis = (
                torch.from_numpy(
                    librosa_mel_fn(
                        self.sampling_rate,
                        self.n_fft,
                        self.config.audio.num_mels,
                        self.config.audio.mel_fmin,
                        self.config.audio.mel_fmax,
                    )
                )
                .float()
                .to(y.device)
            )
            self.mel_bases[key] = mel_basis

        hann_window = self.hann_windows.get(key)
        if hann_window is None:
            hann_window = torch.hann_window(self.config.audio.win_length).to(y.device)
            self.hann_windows[key] = hann_window

        # ---------------------------------------------------------------------

        y = torch.nn.functional.pad(
            y.unsqueeze(1),
            (
                int((self.n_fft - self.hop_length) / 2),
                int((self.n_fft - self.hop_length) / 2),
            ),
            mode="reflect",
        )
        y = y.squeeze(1)

        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=hann_window,
            center=center,
            pad_mode="reflect",
            normalized=False,
            onesided=True,
            return_complex=False,
        )

        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))

        spec = torch.matmul(mel_basis, spec)
        spec = spectral_normalize_torch(spec)

        return spec


# -----------------------------------------------------------------------------


def dynamic_range_compression(x, C=1, clip_val=1e-5):
    return np.log(np.clip(x, a_min=clip_val, a_max=None) * C)


def dynamic_range_decompression(x, C=1):
    return np.exp(x) / C


def dynamic_range_compression_torch(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def dynamic_range_decompression_torch(x, C=1):
    return torch.exp(x) / C


def spectral_normalize_torch(magnitudes):
    output = dynamic_range_compression_torch(magnitudes)
    return output
