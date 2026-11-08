import numpy as np
import pytest

from music_genre.audio.processing import AudioBuffer, segment_audio
from music_genre.evaluation.robustness import compare_condition, perturb_audio


def test_identity_and_minus_six_db_preserve_source_without_clipping():
    source = np.array([0., .5, -1., 3.], dtype=np.float32)
    original = source.copy()
    identity = perturb_audio(source, "identity", "000001")
    gain = perturb_audio(source, "gain_minus_6_db", "000001")
    np.testing.assert_array_equal(source, original)
    np.testing.assert_array_equal(identity, original)
    assert not np.shares_memory(identity, source)
    np.testing.assert_allclose(gain, [0., .2505936168, -.5011872336, 1.5035617009], atol=1e-7)
    assert gain.dtype == np.float32


def test_noise_has_measured_twenty_db_snr_and_reproducible_track_seed():
    signal = np.full(16000, .25, dtype=np.float32)
    noisy = perturb_audio(signal, "white_noise_snr_20_db", "000001")
    noise = noisy.astype(np.float64) - signal
    measured = 10 * np.log10(np.mean(signal.astype(np.float64) ** 2) / np.mean(noise ** 2))
    assert measured == pytest.approx(20., abs=1e-5)
    np.testing.assert_array_equal(noisy, perturb_audio(signal, "white_noise_snr_20_db", "000001"))
    assert not np.array_equal(noisy, perturb_audio(signal, "white_noise_snr_20_db", "000002"))


@pytest.mark.parametrize("samples", [[], [np.nan], [np.inf], [[.1, .2]], [0., 0.]])
def test_noise_rejects_unusable_signals(samples):
    with pytest.raises(ValueError):
        perturb_audio(samples, "white_noise_snr_20_db", "000001")


def test_noise_precedes_segmentation_and_leaves_padding_zero():
    changed = perturb_audio(np.full(8000, .1, dtype=np.float32), "white_noise_snr_20_db", "000001")
    windows = segment_audio(AudioBuffer(changed, 16000, 16000, 1))
    assert len(windows) == 1
    assert windows[0].valid_samples == 8000
    assert np.count_nonzero(windows[0].samples[8000:]) == 0


def test_review_trigger_can_be_flip_rate_without_f1_drop_trigger():
    baseline = np.array([[.8, .2], [.8, .2], [.2, .8], [.2, .8]])
    changed = baseline.copy()
    changed[0] = [.1, .9]
    result = compare_condition([0, 0, 1, 1], baseline, changed, ["A", "B"],
                               {"macro_f1_drop_at_least": .3, "prediction_flip_fraction_at_least": .25})
    assert result["macro_f1_drop"] == pytest.approx(4 / 15)
    assert result["flipped_tracks"] == 1
    assert result["flip_fraction"] == .25
    assert result["review_triggers"] == {"macro_f1_drop": False, "flip_fraction": True}
