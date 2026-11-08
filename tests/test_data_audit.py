import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from music_genre.data.audit import audit_audio, classify_row, duplicate_groups, parse_decoder_log

SETTINGS = json.loads((Path(__file__).resolve().parents[1] / "configs/audit.json").read_text())


def test_exact_pcm_identity_independent_of_container(tmp_path):
    signal = np.linspace(-0.25, 0.25, 16000, dtype=np.float32)
    paths = [tmp_path / "one.wav", tmp_path / "two.wav"]
    sf.write(paths[0], signal, 16000, subtype="FLOAT", format="WAV")
    sf.write(paths[1], signal, 16000, subtype="FLOAT", format="WAVEX")
    a, b = [audit_audio(path, SETTINGS) for path in paths]
    assert a["source_sha256"] != b["source_sha256"]
    assert a["pcm_sha256"] == b["pcm_sha256"]
    assert a["duration_seconds"] == 1
    assert a["decoded_frames"] == 16000


def test_silence_and_corruption_excluded_but_short_audio_reviewed(tmp_path):
    for name, signal, status in [("silent", np.zeros(16000), "excluded"),
                                  ("short", np.full(16000, 0.2), "needs_review")]:
        path = tmp_path / f"{name}.wav"
        sf.write(path, signal, 16000)
        row = audit_audio(path, SETTINGS)
        classify_row(row)
        assert row["status"] == status
    broken = tmp_path / "broken.mp3"
    broken.write_bytes(b"not audio")
    row = audit_audio(broken, SETTINGS)
    classify_row(row)
    assert row["status"] == "excluded"
    assert row["exclusion_reason"] == "decode_failed"
    assert not row["pcm_sha256"]


def test_native_stereo_metrics_do_not_cancel_antiphase(tmp_path):
    path = tmp_path / "stereo.wav"
    sf.write(path, np.tile([0.2, -0.2], (16000, 1)), 16000, subtype="FLOAT")
    row = audit_audio(path, SETTINGS)
    assert row["silent_frame_fraction"] == 0
    assert row["rms_dbfs"] == pytest.approx(20 * np.log10(0.2), abs=0.001)


def test_duplicate_group_cross_split_and_empty_hash_ignored():
    rows = [{"track_id": "1", "split": "train", "hash": "abc"},
            {"track_id": "2", "split": "test", "hash": "abc"},
            {"track_id": "3", "split": "train", "hash": ""},
            {"track_id": "4", "split": "test", "hash": ""}]
    groups = duplicate_groups(rows, "hash")
    assert len(groups) == 1
    assert groups[0]["cross_split"]
    assert groups[0]["track_ids"] == ["1", "2"]


def test_log_attribution_and_unattributed_error():
    text = "AUDIT_BEGIN 000001\nnative decoder warning\nAUDIT_END 000001\n"
    assert parse_decoder_log(text) == {"000001": "native decoder warning"}
    with pytest.raises(ValueError):
        parse_decoder_log("warning before any track")
    with pytest.raises(ValueError):
        parse_decoder_log("AUDIT_BEGIN 000001\n")


@pytest.mark.parametrize("kind", ["empty", "nonfinite", "too_long"])
def test_unusable_audio_has_no_pcm_identity(tmp_path, kind):
    path = tmp_path / "unusable.wav"
    signals = {"empty": np.array([], dtype=np.float32),
               "nonfinite": np.full(10, np.nan, dtype=np.float32),
               "too_long": np.zeros(16000, dtype=np.float32)}
    sf.write(path, signals[kind], 16000, subtype="FLOAT")
    row = audit_audio(path, {**SETTINGS, "max_seconds": 0.5})
    assert row["decode_status"] == "failed"
    assert not row["pcm_sha256"]
    assert row["source_sha256"]


def test_full_scale_is_review_flag_not_automatic_exclusion(tmp_path):
    path = tmp_path / "loud.wav"
    sf.write(path, np.ones(16000), 16000, subtype="FLOAT")
    row = audit_audio(path, SETTINGS)
    classify_row(row)
    assert row["near_full_scale_fraction"] == 1
    assert "near_full_scale" in row["flags"]
    assert row["status"] == "needs_review"
