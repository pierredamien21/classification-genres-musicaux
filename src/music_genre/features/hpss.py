"""Original/harmonic/percussive log-Mel channels from one linear-frequency STFT."""

from pathlib import Path

import librosa
import numpy as np
import torch

from music_genre.audio.processing import AudioWindow
from music_genre.data.audit import sha256_file
from music_genre.models.vit_channels import FrozenViTChannels


def hpss_log_mel(window: AudioWindow, settings: dict, separation: dict) -> np.ndarray:
    if window.samples.ndim != 1 or not np.isfinite(window.samples).all():
        raise ValueError("Expected finite mono audio")
    spectrum = librosa.stft(
        window.samples, n_fft=settings["n_fft"], hop_length=settings["hop_length"],
        window=settings["window"], center=settings["center"],
    )
    harmonic, percussive = librosa.decompose.hpss(
        spectrum, kernel_size=tuple(separation["kernel_size"]),
        power=separation["mask_power"], margin=tuple(separation["margin"]), mask=False,
    )
    channels = []
    for component in (spectrum, harmonic, percussive):
        mel = librosa.feature.melspectrogram(
            S=np.abs(component) ** settings["power"], sr=window.sample_rate,
            n_fft=settings["n_fft"], n_mels=settings["n_mels"], fmin=settings["f_min"],
            fmax=window.sample_rate / 2, htk=False, norm="slaney",
        )
        channels.append(librosa.power_to_db(mel, ref=1.0, amin=settings["amin"], top_db=None))
    result = np.asarray(channels, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite HPSS log-Mel representation")
    return result


class MusicalHpssViT:
    def __init__(self, config: dict, root: Path):
        self.config = config
        if (config["channels"] != "original_harmonic_percussive"
                or config["pooling"] != "last_block_mean_patch_tokens_exclude_prefix"
                or config["hpss"]["domain"] != "complex_stft_before_mel"
                or config["logmel"]["power"] != 2.0):
            raise ValueError("Unsupported HPSS representation protocol")
        torch.set_num_threads(config["threads_per_worker"])
        directory = (root / config["checkpoint_directory"]).resolve()
        if not directory.is_relative_to((root / "artifacts/pretrained").resolve()):
            raise ValueError("Checkpoint outside pretrained directory")
        for filename, expected in config["checkpoint"]["sha256"].items():
            if sha256_file(directory / filename) != expected:
                raise ValueError(f"Pinned checkpoint changed: {filename}")
        self.encoder = FrozenViTChannels(directory)
        self.names = [f"vit_patch_mean_{i:03d}" for i in range(192)]
        self.families = ["vit"] * len(self.names)

    def spectrograms(self, window):
        if window.sample_rate != self.config["sample_rate"]:
            raise ValueError("Unexpected audio sample rate")
        if len(window.samples) != round(window.sample_rate * self.config["window_seconds"]):
            raise ValueError("Expected fixed-duration padded window")
        return hpss_log_mel(window, self.config["logmel"], self.config["hpss"])

    def extract(self, window):
        return self.encoder.extract(
            self.spectrograms(window), self.config["db_min"], self.config["db_max"]
        )
