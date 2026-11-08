"""Original/harmonic/percussive log-Mel channels from one linear-frequency STFT."""

import librosa
import numpy as np

from music_genre.audio.processing import AudioWindow


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
