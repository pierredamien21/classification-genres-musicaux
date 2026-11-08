import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from music_genre.data.audit import sha256_file
from music_genre.inference.predict import CandidatePredictor, validate_audio_file

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def predictor():
    if not (ROOT / "reports/PREPARATION_EVALUATION_FINALE.json").exists():
        pytest.skip("Local prepared export required; no download")
    return CandidatePredictor(ROOT)


def test_short_stereo_file_returns_one_real_window_and_ordered_scores(tmp_path, predictor):
    path = tmp_path / "short.wav"
    tone = .1 * np.sin(2 * np.pi * 440 * np.arange(4000) / 8000)
    sf.write(path, np.column_stack((tone, tone)), 8000, subtype="FLOAT")
    result = predictor.predict_file(path, mode="fma")
    assert result["labels"] == ["Electronic", "Experimental", "Folk", "Hip-Hop",
                                "Instrumental", "International", "Pop", "Rock"]
    assert result["windows"][0]["valid_seconds"] == .5
    assert result["windows"][0]["padded_seconds"] == 9.5
    assert len(result["windows"]) == 1
    assert len(result["scores"]) == 8
    assert np.isfinite(result["scores"]).all()
    assert sum(result["scores"]) == pytest.approx(1)
    np.testing.assert_array_equal(result["scores"], result["windows"][0]["scores"])
    assert len(result["top3"]) == 3
    assert result["top3"][0]["label"] == result["labels"][np.argmax(result["scores"])]
    assert len(result["model_id"]) == 64
    json.dumps(result, allow_nan=False)


def test_silent_file_is_rejected_instead_of_inventing_a_genre(tmp_path, predictor):
    path = tmp_path / "silent.wav"
    sf.write(path, np.zeros(1600), 16000)
    with pytest.raises(ValueError, match="silent"):
        predictor.predict_file(path)


@pytest.mark.parametrize("limit,value,message", [
    ("max_file_bytes", 100, "file size"),
    ("max_seconds", .05, "duration"),
    ("max_pcm_bytes", 100, "PCM"),
])
def test_audio_limits_reject_before_decoding(tmp_path, limit, value, message):
    path = tmp_path / "bounded.wav"
    sf.write(path, np.ones((1600, 2)) * .1, 16000)
    with pytest.raises(ValueError, match=message):
        validate_audio_file(path, **{limit: value})


@pytest.mark.parametrize("kind", ["corrupt", "empty", "nonfinite", "missing", "antiphase"])
def test_public_inference_rejects_unusable_audio(tmp_path, predictor, kind):
    path = tmp_path / "bad.wav"
    if kind == "corrupt":
        path.write_bytes(b"invalid audio")
    elif kind == "empty":
        sf.write(path, np.zeros(0), 16000)
    elif kind == "nonfinite":
        sf.write(path, np.full(1600, np.nan), 16000, subtype="FLOAT")
    elif kind == "antiphase":
        sf.write(path, np.tile([.1, -.1], (1600, 1)), 16000, subtype="FLOAT")
    with pytest.raises(ValueError):
        predictor.predict_file(path)


@pytest.mark.parametrize("kind", ["changed_model", "missing_model", "class_order", "escaped_run"])
def test_loading_rejects_inconsistent_export_before_inference(tmp_path, kind):
    source = ROOT / "reports/PREPARATION_EVALUATION_FINALE.json"
    if not source.exists():
        pytest.skip("Local export metadata required")
    prepared = json.loads(source.read_text("utf-8"))
    pin = prepared["runs"]["vit_block6_hpss"]
    run = json.loads((ROOT / pin["run"]["folder"] / "report.json").read_text("utf-8"))
    for relative in (pin["model_path"], pin["run"]["folder"] + "/report.json",
                     run["feature_cache"] + "/report.json"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    model = tmp_path / pin["model_path"]
    if kind == "changed_model":
        model.write_bytes(b"modified")
    elif kind == "missing_model":
        model.unlink()
    elif kind == "escaped_run":
        pin["run"]["folder"] = "../outside"
    elif kind == "class_order":
        with np.load(model, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["classes"] = np.arange(8)[::-1]
        np.savez(model, **arrays)
        pin["model_sha256"] = run["model_sha256"] = sha256_file(model)
        report = tmp_path / pin["run"]["folder"] / "report.json"
        report.write_text(json.dumps(run), encoding="utf-8")
        pin["run"]["report_sha256"] = sha256_file(report)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/PREPARATION_EVALUATION_FINALE.json").write_text(json.dumps(prepared), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        CandidatePredictor(tmp_path)
