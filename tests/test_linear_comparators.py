import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from threadpoolctl import threadpool_limits

from music_genre.evaluation.window_geometry import window_sample_geometry
from music_genre.features.fusion import AlignedFusion
from music_genre.inference.predict import Predictor

ROOT = Path(__file__).resolve().parents[1]


def test_cached_padding_roundoff_preserves_exact_sample_geometry():
    cached = [{"start_seconds": "20.0", "valid_seconds": "9.976625", "padded_seconds": "0.0233749999999997"}]
    actual = [{"start_seconds": 20., "valid_seconds": 9.976625, "padded_seconds": .023375}]
    assert window_sample_geometry(cached, 16000) == window_sample_geometry(actual, 16000) == [(320000, 159626, 374)]
    actual[0]["padded_seconds"] += 1 / 16000
    assert window_sample_geometry(cached, 16000) != window_sample_geometry(actual, 16000)


@pytest.mark.parametrize("value", [-1., .5 / 16000])
def test_invalid_window_geometry_is_rejected(value):
    with pytest.raises(ValueError):
        window_sample_geometry([{"start_seconds": value, "valid_seconds": 10., "padded_seconds": 0.}], 16000)


@pytest.mark.parametrize("system", ["descriptor", "vit", "fusion", "vit_hpss"])
def test_comparator_reproduces_saved_validation_prediction(system):
    path = ROOT / "reports/PREPARATION_EVALUATION_FINALE.json"
    if not path.exists():
        pytest.skip("Local prepared exports required")
    prepared = json.loads(path.read_text("utf-8"))
    with (ROOT / prepared["sample"]["path"]).open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    pin = prepared["runs"][system]
    with (ROOT / pin["run"]["folder"] / "validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
        expected = next(r for r in csv.DictReader(stream) if r["track_id"] == row["track_id"])
    result = Predictor(ROOT, system).predict_file(ROOT / "data/raw" / row["audio_path"])
    assert result["labels"] == prepared["labels"]
    np.testing.assert_allclose(result["scores"], [float(expected[f"p_{label}"]) for label in prepared["labels"]],
                               atol=1e-6, rtol=0)


@pytest.mark.parametrize("kind", ["columns", "order", "windows"])
def test_fusion_refuses_misaligned_representation_configuration(kind):
    path = ROOT / "reports/PREPARATION_EVALUATION_FINALE.json"
    if not path.exists():
        pytest.skip("Local feature metadata required")
    prepared = json.loads(path.read_text("utf-8"))
    report = json.loads((ROOT / prepared["runs"]["fusion"]["feature_cache"] / "report.json").read_text("utf-8"))
    config = report["provenance"]["config"]
    if kind == "columns":
        config["descriptor_columns"] = config["descriptor_columns"][::-1]
    elif kind == "order":
        config["order"] = config["order"][::-1]
    else:
        config["descriptors"]["offsets_seconds"] = [10.]
    with pytest.raises(ValueError, match="Fusion"):
        AlignedFusion(config, ROOT)


@pytest.mark.parametrize("kind", ["model", "feature_report", "escaped_run"])
def test_loading_rejects_changed_or_escaped_comparator(tmp_path, kind):
    path = ROOT / "reports/PREPARATION_EVALUATION_FINALE.json"
    if not path.exists():
        pytest.skip("Local export metadata required")
    prepared = json.loads(path.read_text("utf-8"))
    pin = prepared["runs"]["descriptor"]
    for relative in (pin["model_path"], pin["run"]["folder"] + "/report.json",
                     pin["feature_cache"] + "/report.json"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    if kind == "model":
        (tmp_path / pin["model_path"]).write_bytes(b"changed")
    elif kind == "feature_report":
        (tmp_path / pin["feature_cache"] / "report.json").write_text("{}")
    else:
        pin["run"]["folder"] = "../elsewhere"
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/PREPARATION_EVALUATION_FINALE.json").write_text(json.dumps(prepared), encoding="utf-8")
    with pytest.raises(ValueError):
        Predictor(tmp_path, "descriptor")


@pytest.mark.parametrize("system", ["descriptor", "vit", "fusion", "vit_hpss"])
def test_short_stereo_audio_has_one_vote_and_silence_is_rejected(tmp_path, system):
    if not (ROOT / "reports/PREPARATION_EVALUATION_FINALE.json").exists():
        pytest.skip("Local prepared exports required")
    predictor = Predictor(ROOT, system)
    tone = .1 * np.sin(2 * np.pi * 440 * np.arange(4000) / 8000)
    path = tmp_path / "short.wav"
    sf.write(path, np.column_stack((tone, tone)), 8000, subtype="FLOAT")
    result = predictor.predict_file(path)
    assert len(result["windows"]) == 1
    assert result["windows"][0]["valid_seconds"] == .5
    assert result["windows"][0]["padded_seconds"] == 9.5
    np.testing.assert_array_equal(result["scores"], result["windows"][0]["scores"])
    assert sum(result["scores"]) == pytest.approx(1)
    sf.write(path, np.zeros(8000), 8000)
    with pytest.raises(ValueError, match="silent"):
        predictor.predict_file(path)


def test_vit_inference_matches_cached_windows_with_multithreaded_blas_caller():
    path = ROOT / "reports/PREPARATION_EVALUATION_FINALE.json"
    if not path.exists():
        pytest.skip("Local feature cache required")
    prepared = json.loads(path.read_text("utf-8"))
    with (ROOT / prepared["sample"]["path"]).open(encoding="utf-8", newline="") as stream:
        row = next(r for r in csv.DictReader(stream) if r["track_id"] == "128484")
    cache = ROOT / prepared["runs"]["vit"]["feature_cache"]
    with (cache / "index.csv").open(encoding="utf-8", newline="") as stream:
        positions = [i for i, r in enumerate(csv.DictReader(stream)) if r["track_id"] == row["track_id"]]
    predictor = Predictor(ROOT, "vit")
    matrix = np.load(cache / "features.npy", mmap_mode="r", allow_pickle=False)
    expected = predictor.model.predict_proba(matrix[positions])
    with threadpool_limits(limits=8, user_api="blas"):
        result = predictor.predict_file(ROOT / "data/raw" / row["audio_path"])
    np.testing.assert_allclose([w["scores"] for w in result["windows"]], expected, atol=1e-6, rtol=0)
