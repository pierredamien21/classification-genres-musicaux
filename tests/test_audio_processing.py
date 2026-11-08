import tomllib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from music_genre.audio.processing import AudioBuffer, load_audio, segment_audio
from music_genre.features.logmel import log_mel

ROOT = Path(__file__).resolve().parents[1]


def test_wav_stereo_downmix_and_resample(tmp_path):
    path = tmp_path / "stereo.wav"
    samples = np.tile(np.array([0.25, 0.75], dtype=np.float32), (8000, 1))
    sf.write(path, samples, 8000, subtype="FLOAT")
    audio = load_audio(path, sample_rate=16000)
    assert audio.source_channels == 2
    assert audio.source_sample_rate == 8000
    assert audio.samples.shape == (16000,)
    # Ignore filter transients at the two boundaries of the finite signal.
    np.testing.assert_allclose(audio.samples[500:-500], 0.5, atol=1e-4)


def test_short_audio_does_not_create_empty_or_duplicate_votes():
    audio = AudioBuffer(np.ones(8000, dtype=np.float32), 16000, 16000, 1)
    windows = segment_audio(audio, (0, 0, 10, 20))
    assert len(windows) == 1
    assert windows[0].valid_samples == 8000
    np.testing.assert_array_equal(windows[0].samples[:8000], 1)
    np.testing.assert_array_equal(windows[0].samples[8000:], 0)


def test_window_boundaries_preserve_order_without_overlap():
    samples = np.repeat(np.array([1, 2, 3], dtype=np.float32), 160000)
    audio = AudioBuffer(samples, 16000, 16000, 1)
    windows = segment_audio(audio)
    assert len(windows) == 3
    for i, window in enumerate(windows):
        assert window.valid_samples == 160000
        assert window.start_seconds == 10 * i
        np.testing.assert_array_equal(window.samples, i + 1)


@pytest.mark.parametrize("start", [-1, float("nan"), float("inf")])
def test_invalid_offsets_fail(start):
    audio = AudioBuffer(np.ones(10, dtype=np.float32), 16000, 16000, 1)
    with pytest.raises(ValueError):
        segment_audio(audio, (start,))


@pytest.mark.parametrize("kind", ["empty", "nonfinite", "corrupt", "too_long"])
def test_invalid_audio_fails_explicitly(tmp_path, kind):
    path = tmp_path / "invalid.wav"
    if kind == "corrupt":
        path.write_bytes(b"this is not an audio file")
    else:
        samples = {"empty": np.array([], dtype=np.float32),
                   "nonfinite": np.full(100, np.nan, dtype=np.float32),
                   "too_long": np.zeros(16000, dtype=np.float32)}[kind]
        sf.write(path, samples, 16000, subtype="FLOAT")
    with pytest.raises(ValueError):
        load_audio(path, max_seconds=0.5)


def test_silence_has_finite_logmel_and_remains_observable(tmp_path):
    path = tmp_path / "silence.wav"
    sf.write(path, np.zeros(16000), 16000)
    window = segment_audio(load_audio(path))[0]
    config = tomllib.loads((ROOT / "configs/diagnostic.toml").read_text(encoding="utf-8"))
    mel = log_mel(window, config["logmel"])
    assert mel.shape[0] == 128
    assert np.isfinite(mel).all()
    assert np.all(mel < -90)
