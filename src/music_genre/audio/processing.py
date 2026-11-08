"""Bounded decoding and deterministic windows for the initial audio diagnostic."""

import math
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioBuffer:
    samples: np.ndarray
    sample_rate: int
    source_sample_rate: int
    source_channels: int


@dataclass(frozen=True)
class AudioWindow:
    samples: np.ndarray
    sample_rate: int
    start_seconds: float
    valid_samples: int


def load_audio(path: Path, sample_rate: int = 16000, max_seconds: float = 300) -> AudioBuffer:
    """Decode at most the configured duration, then downmix and resample.

    Silence is retained for diagnosis, not silently discarded as a dataset error.
    """
    if sample_rate <= 0 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("Sample rate and maximum duration must be positive")
    try:
        with sf.SoundFile(path) as source:
            source_rate, channels = source.samplerate, source.channels
            limit = int(max_seconds * source_rate)
            if source.frames > limit:
                raise ValueError(f"Audio exceeds the {max_seconds}s limit")
            if channels > 8:
                raise ValueError("Audio with more than eight channels is not supported")
            samples = source.read(frames=limit + 1, dtype="float32", always_2d=True)
    except (sf.LibsndfileError, OSError) as error:
        raise ValueError(f"Cannot decode audio: {path.name}") from error
    if len(samples) == 0 or len(samples) > limit or not np.isfinite(samples).all():
        raise ValueError("Audio is empty, too long or contains non-finite samples")
    mono = samples.mean(axis=1, dtype=np.float32)
    if source_rate != sample_rate:
        mono = librosa.resample(
            mono, orig_sr=source_rate, target_sr=sample_rate, res_type="soxr_hq"
        )
    return AudioBuffer(np.ascontiguousarray(mono, dtype=np.float32), sample_rate,
                       source_rate, channels)


def segment_audio(
    audio: AudioBuffer,
    starts_seconds: tuple[float, ...] = (0.0, 10.0, 20.0),
    duration_seconds: float = 10.0,
) -> list[AudioWindow]:
    """Right-pad short windows, omit starts beyond the signal, deduplicate starts."""
    if audio.sample_rate <= 0 or not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("Invalid sample rate or window duration")
    if audio.samples.ndim != 1 or not np.isfinite(audio.samples).all():
        raise ValueError("Expected a finite mono signal")
    length = round(duration_seconds * audio.sample_rate)
    if length < 1:
        raise ValueError("Window must contain at least one sample")
    windows, seen = [], set()
    for start in starts_seconds:
        if not math.isfinite(start) or start < 0:
            raise ValueError("Window starts must be finite and non-negative")
        offset = round(start * audio.sample_rate)
        if offset in seen or offset >= len(audio.samples):
            continue
        seen.add(offset)
        chunk = audio.samples[offset:offset + length]
        padded = np.zeros(length, dtype=np.float32)
        padded[:len(chunk)] = chunk
        windows.append(AudioWindow(padded, audio.sample_rate, offset / audio.sample_rate,
                                   len(chunk)))
    return windows
