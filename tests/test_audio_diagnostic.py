import numpy as np
import pytest

from music_genre.audio.processing import AudioWindow
from music_genre.evaluation.audio_diagnostic import diagnose_window


def test_diagnostic_distinguishes_real_signal_from_padding():
    window = AudioWindow(np.r_[np.full(80000, 0.5, dtype=np.float32), np.zeros(80000, dtype=np.float32)],
                         16000, 20.0, 80000)
    row = {"split": "validation", "start_seconds": "20", "valid_seconds": "5", "padded_seconds": "5"}
    mel = np.full((128, 622), -30, dtype=np.float32)
    result = diagnose_window(window, mel, mel.copy(), row)
    assert result["valid_seconds"] == 5
    assert result["padded_seconds"] == 5
    assert result["valid_rms_dbfs"] == pytest.approx(-6.020599913, abs=1e-6)
    assert result["max_cache_error_db"] == 0
    assert result["below_db_range_fraction"] == 0


@pytest.mark.parametrize("change", ["split", "timing", "cache", "nan"])
def test_diagnostic_rejects_untrusted_or_mismatched_window(change):
    window = AudioWindow(np.ones(160000, dtype=np.float32), 16000, 0.0, 160000)
    row = {"split": "validation", "start_seconds": "0", "valid_seconds": "10", "padded_seconds": "0"}
    mel = np.full((128, 622), -30, dtype=np.float32)
    cached = mel.copy()
    if change == "split":
        row["split"] = "test"
    elif change == "timing":
        row["start_seconds"] = "10"
    elif change == "cache":
        cached[0, 0] = -25
    else:
        cached[0, 0] = np.nan
    with pytest.raises(ValueError):
        diagnose_window(window, mel, cached, row)
