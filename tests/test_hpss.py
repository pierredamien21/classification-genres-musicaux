import json
from pathlib import Path

import numpy as np
import pytest

from music_genre.audio.processing import AudioWindow
from music_genre.features.hpss import hpss_log_mel
from music_genre.features.logmel import log_mel

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/vit_hpss.json").read_text(encoding="utf-8"))


def channels(samples):
    window = AudioWindow(np.asarray(samples, dtype=np.float32), 16000, 0.0, len(samples))
    return window, hpss_log_mel(window, CONFIG["logmel"], CONFIG["hpss"])


def test_original_channel_matches_reference_and_silence_is_finite():
    rng = np.random.default_rng(42)
    window, values = channels(rng.normal(0, 0.1, 160000))
    assert values.shape == (3, 128, 622)
    np.testing.assert_allclose(values[0], log_mel(window, CONFIG["logmel"]), rtol=0, atol=1e-5)
    _, silence = channels(np.zeros(160000))
    assert np.isfinite(silence).all()
    np.testing.assert_allclose(silence, -100.0, atol=1e-4)


def test_stationary_tone_and_impulses_have_expected_channel_dominance():
    time = np.arange(160000) / 16000
    _, tone = channels(0.5 * np.sin(2 * np.pi * 440 * time))
    pulses = np.zeros(160000, dtype=np.float32)
    pulses[8000::16000] = 1.0
    _, impulses = channels(pulses)
    # Compare total linear Mel power, excluding the original channel.
    tone_energy = (10 ** (tone / 10)).sum(axis=(1, 2))
    impulse_energy = (10 ** (impulses / 10)).sum(axis=(1, 2))
    assert tone_energy[1] > 10 * tone_energy[2]
    assert impulse_energy[2] > 10 * impulse_energy[1]


def test_repeated_channels_reproduce_original_vit():
    weights = ROOT / "artifacts/pretrained/vit_tiny"
    if not (weights / "model.safetensors").exists():
        pytest.skip("Local checkpoint required; no download")
    import torch

    from music_genre.models.vit import FrozenViT
    from music_genre.models.vit_channels import FrozenViTChannels

    torch.set_num_threads(2)
    original, multichannel = FrozenViT(weights), FrozenViTChannels(weights)
    mel = np.random.default_rng(42).uniform(-100, 30, (128, 622)).astype(np.float32)
    repeated = np.repeat(mel[None], 3, axis=0)
    np.testing.assert_allclose(multichannel.extract(repeated, -80, 20),
                               original.extract(mel, -80, 20), rtol=0, atol=1e-6)
    assert not multichannel.model.training
    assert not any(p.requires_grad for p in multichannel.model.parameters())
    with pytest.raises(ValueError, match="3-channel"):
        multichannel.extract(mel, -80, 20)
