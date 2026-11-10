import numpy as np
import pytest

from music_genre.audio.degrade import augmentation_for, degrade, mp3_roundtrip

RATE = 16000


def tone(seconds=3.0, rate=RATE):
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def test_white_noise_hits_the_requested_snr_and_is_reproducible_per_track():
    clean = tone()
    noisy = degrade(clean, RATE, {"kind": "white_noise", "snr_db": 10}, "000002", seed=7)
    noise = noisy.astype(np.float64) - clean
    snr = 10 * np.log10(np.mean(clean.astype(np.float64) ** 2) / np.mean(noise ** 2))
    assert snr == pytest.approx(10, abs=1e-3)
    assert np.array_equal(noisy, degrade(clean, RATE, {"kind": "white_noise", "snr_db": 10}, "000002", seed=7))
    assert not np.array_equal(noisy, degrade(clean, RATE, {"kind": "white_noise", "snr_db": 10}, "000005", seed=7))


def test_gain_scales_without_clipping():
    louder = degrade(tone(), RATE, {"kind": "gain", "db": 6}, "000002", seed=7)
    np.testing.assert_allclose(louder, tone() * 10 ** (6 / 20), rtol=1e-6)


def test_silent_signal_has_no_snr():
    with pytest.raises(ValueError):
        degrade(np.zeros(RATE, dtype=np.float32), RATE, {"kind": "white_noise", "snr_db": 20}, "x", seed=7)


def test_mp3_roundtrip_keeps_length_and_level_and_reports_64_kbps():
    rate = 44100
    signal = tone(10.0, rate) * 6  # peak 1.8: encoded after peak normalization, not clipped
    decoded, kbps = mp3_roundtrip(signal, rate, 64)
    assert kbps == pytest.approx(64, abs=1)
    assert decoded.shape == signal.shape and decoded.dtype == np.float32
    middle = slice(rate, -rate)
    error = np.sqrt(np.mean((decoded[middle] - signal[middle]) ** 2) / np.mean(signal[middle] ** 2))
    assert error < 0.1


def test_mp3_resamples_22050_hz_sources_to_44100_and_back():
    decoded, kbps = mp3_roundtrip(tone(10.0, 22050), 22050, 64)
    assert len(decoded) == 22050 * 10 and kbps == pytest.approx(64, abs=1)


def test_augmentation_is_one_reproducible_draw_per_track_inside_the_declared_ranges():
    recipe = {"kinds": ["white_noise", "gain", "mp3"], "snr_db": [10, 30], "gain_db": [-6, 6], "kbps": 64}
    draws = [augmentation_for(f"{i:06d}", recipe, seed=11) for i in range(300)]
    assert draws[5] == augmentation_for("000005", recipe, seed=11)
    assert {d["kind"] for d in draws} == {"white_noise", "gain", "mp3"}
    assert all(10 <= d["snr_db"] <= 30 for d in draws if d["kind"] == "white_noise")
    assert all(-6 <= d["db"] <= 6 for d in draws if d["kind"] == "gain")
    assert all(d == {"kind": "mp3", "kbps": 64} for d in draws if d["kind"] == "mp3")


def test_load_degraded_matches_load_audio_at_zero_gain_and_keeps_length_through_mp3(tmp_path):
    import soundfile as sf

    from music_genre.audio.degrade import load_degraded
    from music_genre.audio.processing import load_audio
    path = tmp_path / "x.wav"
    stereo = np.stack([tone(4.0, 44100), tone(4.0, 44100) * 0.5], axis=1)
    sf.write(path, stereo, 44100)
    clean = load_audio(path, sample_rate=RATE)
    same = load_degraded(path, RATE, {"name": "g0", "seed": 1, "spec": {"kind": "gain", "db": 0}}, "000002")
    assert np.array_equal(same.samples, clean.samples) and same.source_sample_rate == 44100
    mp3 = load_degraded(path, RATE, {"name": "mp3", "seed": 1, "spec": {"kind": "mp3", "kbps": 64}}, "000002")
    assert mp3.samples.shape == clean.samples.shape and not np.array_equal(mp3.samples, clean.samples)
    recipe = {"kinds": ["gain"], "snr_db": [10, 30], "gain_db": [-6, -6], "kbps": 64}
    augmented = load_degraded(path, RATE, {"name": "augmentation", "seed": 3, "recipe": recipe}, "000002")
    np.testing.assert_allclose(augmented.samples, clean.samples * 10 ** (-6 / 20), rtol=1e-5)
