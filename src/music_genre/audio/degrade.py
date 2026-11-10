"""V2 phase 4 audio degradations: white noise at a given SNR, digital gain, MP3 recompression.

Noise and gain act on the model-rate signal, as V1 robustness did (evaluation/robustness.py, pinned
by the V1 lock). MP3 needs the source rate: mp3_roundtrip runs before resampling. Noise draws are
keyed by (seed, track, condition) so every run and worker sees the same signal.
"""

import hashlib
import io

import librosa
import numpy as np
import soundfile as sf

from music_genre.audio.processing import AudioBuffer, load_audio

# libsndfile 1.2.2 (bundled LAME) maps compression_level 0.86-0.89 to 64 kb/s CBR at 44.1/48 kHz;
# the measured bitrate is checked on every call instead of trusting the mapping.
MP3_LEVEL = {64: 0.87}
MP3_RATES = (44100, 48000)


def _rng(seed, *parts):
    key = ":".join(str(p) for p in (seed, *parts)).encode()
    return np.random.Generator(np.random.PCG64(int(hashlib.sha256(key).hexdigest()[:16], 16)))


def degrade(samples, rate, spec, track_id, seed):
    """Apply a white-noise or gain spec to a mono float signal; MP3 goes through mp3_roundtrip."""
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Expected a nonempty finite mono signal")
    if spec["kind"] == "gain":
        return (samples * np.float32(10 ** (spec["db"] / 20))).astype(np.float32)
    if spec["kind"] == "white_noise":
        signal = samples.astype(np.float64)
        power = np.mean(signal ** 2)
        if power == 0:
            raise ValueError("SNR is undefined for a silent signal")
        noise = _rng(seed, track_id, "white_noise", spec["snr_db"]).standard_normal(len(samples))
        noise *= np.sqrt(power / 10 ** (spec["snr_db"] / 10) / np.mean(noise ** 2))
        return (signal + noise).astype(np.float32)
    raise ValueError(f"Unknown degradation for a model-rate signal: {spec['kind']}")


def mp3_roundtrip(samples, rate, kbps):
    """Encode mono audio to constant-bitrate MP3 in memory and decode it back at the same rate.

    The signal is divided by its peak before encoding and multiplied back after decoding, so the
    codec, not clipping, is measured. Sources outside 44.1/48 kHz go through 44.1 kHz.
    Returns (decoded float32 samples, measured kb/s).
    """
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Expected a nonempty finite mono signal")
    codec_rate = rate if rate in MP3_RATES else 44100
    signal = samples if codec_rate == rate else librosa.resample(
        samples, orig_sr=rate, target_sr=codec_rate, res_type="soxr_hq")
    scale = max(1.0, float(np.max(np.abs(signal))) / 0.99)
    buffer = io.BytesIO()
    sf.write(buffer, signal / scale, codec_rate, format="MP3", bitrate_mode="CONSTANT",
             compression_level=MP3_LEVEL[kbps])
    measured = len(buffer.getvalue()) * 8 / (len(signal) / codec_rate) / 1000
    buffer.seek(0)
    decoded, decoded_rate = sf.read(buffer, dtype="float32")
    if decoded_rate != codec_rate or decoded.ndim != 1 or len(decoded) != len(signal):
        raise ValueError("MP3 roundtrip changed the rate, channels or length")
    if abs(measured - kbps) > 2:  # headers inflate short files; neighbouring CBR steps are 8 kb/s away
        raise ValueError(f"MP3 bitrate {measured:.1f} kb/s instead of {kbps}")
    decoded = (decoded * scale).astype(np.float32)
    if codec_rate != rate:
        decoded = librosa.resample(decoded, orig_sr=codec_rate, target_sr=rate, res_type="soxr_hq")
        decoded = np.pad(decoded, (0, max(0, len(samples) - len(decoded))))[:len(samples)]
    return decoded.astype(np.float32), measured


def augmentation_for(track_id, recipe, seed):
    """One training degradation per track: kind uniform, then its parameter uniform in its range."""
    rng = _rng(seed, track_id, "augmentation")
    kind = recipe["kinds"][int(rng.integers(len(recipe["kinds"])))]
    if kind == "white_noise":
        return {"kind": kind, "snr_db": float(rng.uniform(*recipe["snr_db"]))}
    if kind == "gain":
        return {"kind": kind, "db": float(rng.uniform(*recipe["gain_db"]))}
    if kind == "mp3":
        return {"kind": kind, "kbps": recipe["kbps"]}
    raise ValueError(f"Unknown augmentation kind: {kind}")


def load_degraded(path, sample_rate, degradation, track_id):
    """load_audio, then one degradation: {"name", "seed", "spec"} or {"name", "seed", "recipe"}."""
    spec = degradation.get("spec") or augmentation_for(track_id, degradation["recipe"], degradation["seed"])
    if spec["kind"] != "mp3":
        audio = load_audio(path, sample_rate=sample_rate)
        samples = degrade(audio.samples, sample_rate, spec, track_id, degradation["seed"])
        return AudioBuffer(samples, sample_rate, audio.source_sample_rate, audio.source_channels)
    native = load_audio(path, sample_rate=sf.info(path).samplerate)  # mono, not resampled
    decoded, _ = mp3_roundtrip(native.samples, native.sample_rate, spec["kbps"])
    if native.sample_rate != sample_rate:
        decoded = librosa.resample(decoded, orig_sr=native.sample_rate, target_sr=sample_rate, res_type="soxr_hq")
    return AudioBuffer(np.ascontiguousarray(decoded, dtype=np.float32), sample_rate,
                       native.source_sample_rate, native.source_channels)
