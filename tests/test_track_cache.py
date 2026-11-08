import numpy as np
import pytest

from music_genre.features.track_cache import load_track_cache, save_track_cache


def test_track_cache_recovers_completed_windows_without_changing_values(tmp_path):
    target = tmp_path / "000001.npz"
    x = np.arange(576, dtype=np.float32).reshape(3, 192)
    valid = np.array([10., 10., 9.98])
    assert load_track_cache(target, "config-a", "audio-a") is None
    save_track_cache(target, "config-a", "audio-a", x, valid)
    restored, durations = load_track_cache(target, "config-a", "audio-a")
    np.testing.assert_array_equal(restored, x)
    np.testing.assert_array_equal(durations, valid)


@pytest.mark.parametrize("kind", ["checksum", "config", "source"])
def test_resume_refuses_corruption_or_different_identity(tmp_path, kind):
    path = tmp_path / "000001.npz"
    save_track_cache(path, "config-a", "audio-a", np.zeros((3, 192)), np.full(3, 10.))
    if kind == "checksum":
        path.with_suffix(".json").write_text('{"sha256": "wrong"}', encoding="utf-8")
    with pytest.raises(ValueError):
        load_track_cache(path, "config-b" if kind == "config" else "config-a",
                         "audio-b" if kind == "source" else "audio-a")


@pytest.mark.parametrize("kind", ["shape", "nan", "duration"])
def test_incomplete_or_invalid_windows_cannot_be_published(tmp_path, kind):
    x, valid = np.zeros((3, 192)), np.full(3, 10.)
    if kind == "shape":
        x = x[:2]
    elif kind == "nan":
        x[0, 0] = np.nan
    else:
        valid[2] = 0
    with pytest.raises(ValueError):
        save_track_cache(tmp_path / "000001.npz", "config", "audio", x, valid)
