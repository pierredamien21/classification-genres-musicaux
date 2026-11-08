"""Explicit log-Mel settings used for diagnostics, not a tuned representation."""

import librosa
import numpy as np

from music_genre.audio.processing import AudioWindow


def log_mel(window: AudioWindow, settings: dict) -> np.ndarray:
    power = librosa.feature.melspectrogram(
        y=window.samples,
        sr=window.sample_rate,
        n_fft=settings["n_fft"],
        hop_length=settings["hop_length"],
        n_mels=settings["n_mels"],
        fmin=settings["f_min"],
        fmax=window.sample_rate / 2,
        window=settings["window"],
        center=settings["center"],
        power=settings["power"],
        htk=False,
        norm="slaney",
    )
    result = librosa.power_to_db(power, ref=1.0, amin=settings["amin"], top_db=None)
    if not np.isfinite(result).all():
        raise ValueError("Non-finite log-Mel representation")
    return np.asarray(result, dtype=np.float32)
