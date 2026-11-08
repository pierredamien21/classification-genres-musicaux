"""Explicit timbre, harmonic pitch-class and rhythmic periodicity descriptors."""

import librosa
import numpy as np
from scipy.signal import fftconvolve

from music_genre.audio.processing import AudioWindow


class MusicalDescriptors:
    def __init__(self, config):
        self.config = config
        self.sr = config["sample_rate"]
        self.n_fft = config["n_fft"]
        self.hop = config["hop_length"]
        self.mel_filter = librosa.filters.mel(
            sr=self.sr, n_fft=self.n_fft, n_mels=config["n_mels"],
            fmin=0, fmax=self.sr / 2, htk=False, norm="slaney",
        )
        self.chroma_filter = librosa.filters.chroma(
            sr=self.sr, n_fft=self.n_fft, n_chroma=12, tuning=0
        )
        self.bpms = np.arange(config["bpm_min"], config["bpm_max"] + 1, config["bpm_step"])
        self.names, self.families = [], []
        for name, size, family in (
            ("mfcc", config["n_mfcc"], "timbre"), ("centroid", 1, "timbre"),
            ("bandwidth", 1, "timbre"),
            ("contrast", config["contrast_n_bands"] + 1, "timbre"),
            ("rms", 1, "timbre"), ("chroma", 12, "harmony"),
        ):
            for statistic in ("mean", "std"):
                for i in range(size):
                    self.names.append(f"{name}_{i:02d}_{statistic}")
                    self.families.append(family)
        for bpm in self.bpms:
            self.names.append(f"onset_acf_{bpm:03d}_bpm")
            self.families.append("rhythm")

    def extract(self, window: AudioWindow) -> np.ndarray:
        if window.sample_rate != self.sr:
            raise ValueError("Descriptor sample rate mismatch")
        if not 0 < window.valid_samples <= len(window.samples):
            raise ValueError("Invalid number of real audio samples")
        if window.samples.ndim != 1 or not np.isfinite(window.samples).all():
            raise ValueError("Expected finite mono audio")
        signal = np.asarray(window.samples[:window.valid_samples], dtype=np.float32)
        if len(signal) < self.n_fft:
            signal = np.pad(signal, (0, self.n_fft - len(signal)))
        spectrum = librosa.stft(
            signal, n_fft=self.n_fft, hop_length=self.hop, window="hann", center=False
        )
        magnitude = np.abs(spectrum)
        power = magnitude ** 2
        logmel = librosa.power_to_db(
            self.mel_filter @ power, ref=1.0, amin=1e-10, top_db=None
        )
        centroid = librosa.feature.spectral_centroid(S=magnitude, sr=self.sr)
        chroma = librosa.util.normalize(self.chroma_filter @ power, norm=np.inf, axis=0)
        blocks = [
            librosa.feature.mfcc(S=logmel, n_mfcc=self.config["n_mfcc"],
                                 dct_type=2, norm="ortho", lifter=0),
            centroid,
            librosa.feature.spectral_bandwidth(S=magnitude, sr=self.sr, centroid=centroid, p=2),
            librosa.feature.spectral_contrast(
                S=magnitude, sr=self.sr, n_fft=self.n_fft,
                fmin=self.config["contrast_fmin"], n_bands=self.config["contrast_n_bands"],
                quantile=self.config["contrast_quantile"], linear=False,
            ),
            librosa.feature.rms(y=signal, frame_length=self.n_fft,
                                hop_length=self.hop, center=False),
            chroma,
        ]
        summaries = [summary for block in blocks
                     for summary in (block.mean(axis=1), block.std(axis=1, ddof=0))]
        flux = np.maximum(np.diff(logmel, axis=1, prepend=logmel[:, :1]), 0).mean(axis=0)
        centered = flux.astype(np.float64) - flux.mean(dtype=np.float64)
        energy = float(centered @ centered)
        if energy > 1e-20 and len(centered) > 1:
            acf = fftconvolve(centered, centered[::-1], mode="full")[len(centered) - 1:] / energy
            lags = 60 * self.sr / (self.hop * self.bpms)
            rhythm = np.interp(lags, np.arange(len(acf)), acf, left=0, right=0)
        else:
            rhythm = np.zeros(len(self.bpms))
        result = np.concatenate([*summaries, rhythm]).astype(np.float32)
        if result.shape != (len(self.names),) or not np.isfinite(result).all():
            raise ValueError("Invalid descriptor vector")
        return result
