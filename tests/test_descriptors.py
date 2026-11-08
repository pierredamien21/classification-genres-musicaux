import json
from pathlib import Path

import numpy as np
import pytest

from music_genre.audio.processing import AudioWindow
from music_genre.features.descriptors import MusicalDescriptors

CONFIG = json.loads((Path(__file__).resolve().parents[1] / "configs/descriptors.json").read_text())


@pytest.fixture(scope="module")
def extractor():
    return MusicalDescriptors(CONFIG)


def test_sine_timbre_and_chroma_are_physical(extractor):
    rate = CONFIG["sample_rate"]
    signal = (0.3 * np.sin(2 * np.pi * 440 * np.arange(rate * 2) / rate)).astype(np.float32)
    values = dict(zip(extractor.names, extractor.extract(AudioWindow(signal, rate, 0, len(signal)))))
    assert values["centroid_00_mean"] == pytest.approx(440, abs=3)
    assert values["rms_00_mean"] == pytest.approx(0.3 / np.sqrt(2), abs=0.003)
    assert max(range(12), key=lambda i: values[f"chroma_{i:02d}_mean"]) == 9
    assert len(values) == 125


def test_short_padding_does_not_change_real_audio_statistics(extractor):
    signal = np.random.default_rng(7).normal(0, 0.1, 8000).astype(np.float32)
    padded = np.pad(signal, (0, 152000))
    a = extractor.extract(AudioWindow(signal, 16000, 0, 8000))
    b = extractor.extract(AudioWindow(padded, 16000, 0, 8000))
    np.testing.assert_array_equal(a, b)


def test_silence_and_very_short_audio_remain_finite(extractor):
    for length in (1, 16000):
        x = extractor.extract(AudioWindow(np.zeros(length, dtype=np.float32), 16000, 0, length))
        assert np.isfinite(x).all()
        np.testing.assert_array_equal(x[-41:], 0)


def test_invalid_audio_is_rejected(extractor):
    with pytest.raises(ValueError):
        extractor.extract(AudioWindow(np.array([np.nan]), 16000, 0, 1))
    with pytest.raises(ValueError):
        extractor.extract(AudioWindow(np.zeros(16000), 8000, 0, 16000))


def test_rhythmic_pulses_create_expected_periodicity(extractor):
    signal = np.zeros(160000, dtype=np.float32)
    burst = np.random.default_rng(4).normal(0, 0.2, 800).astype(np.float32) * np.hanning(800)
    for start in range(0, len(signal) - len(burst), 8000):
        signal[start:start + len(burst)] = burst
    values = dict(zip(extractor.names, extractor.extract(AudioWindow(signal, 16000, 0, len(signal)))))
    assert values["onset_acf_120_bpm"] > 0.7
    assert values["onset_acf_120_bpm"] > values["onset_acf_095_bpm"] + 0.5
