import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from music_genre.inference.predict import Predictor

ROOT = Path(__file__).resolve().parents[1]


def test_original_block6_reproduces_saved_validation_track():
    preparation = ROOT / "reports/PREPARATION_EVALUATION_FINALE.json"
    if not preparation.exists():
        pytest.skip("Local prepared export required; no download")
    prepared = json.loads(preparation.read_text("utf-8"))
    with (ROOT / prepared["sample"]["path"]).open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    pin = prepared["runs"]["vit_block6"]
    with (ROOT / pin["run"]["folder"] / "validation_predictions.csv").open(
            encoding="utf-8", newline="") as stream:
        expected = next(r for r in csv.DictReader(stream) if r["track_id"] == row["track_id"])
    result = Predictor(ROOT, "vit_block6").predict_file(ROOT / "data/raw" / row["audio_path"])
    assert result["labels"] == prepared["labels"]
    np.testing.assert_allclose(result["scores"],
                               [float(expected[f"p_{label}"]) for label in prepared["labels"]],
                               atol=1e-6, rtol=0)
    assert [w["start_seconds"] for w in result["windows"]] == [0., 10., 20.]


@pytest.mark.parametrize("kind", ["changed_model", "missing_model", "escaped_run"])
def test_original_adapter_rejects_untrusted_export(tmp_path, kind):
    source = ROOT / "reports/PREPARATION_EVALUATION_FINALE.json"
    if not source.exists():
        pytest.skip("Local export metadata required")
    prepared = json.loads(source.read_text("utf-8"))
    pin = prepared["runs"]["vit_block6"]
    for relative in (pin["model_path"], pin["run"]["folder"] + "/report.json",
                     pin["feature_cache"] + "/report.json"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    if kind == "changed_model":
        (tmp_path / pin["model_path"]).write_bytes(b"modified")
    elif kind == "missing_model":
        (tmp_path / pin["model_path"]).unlink()
    else:
        pin["run"]["folder"] = "../outside"
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/PREPARATION_EVALUATION_FINALE.json").write_text(
        json.dumps(prepared), encoding="utf-8")
    with pytest.raises((ValueError, FileNotFoundError)):
        Predictor(tmp_path, "vit_block6")


def test_original_adapter_short_file_has_no_duplicate_votes(tmp_path):
    if not (ROOT / "reports/PREPARATION_EVALUATION_FINALE.json").exists():
        pytest.skip("Local prepared export required")
    path = tmp_path / "short.wav"
    sf.write(path, .1 * np.sin(2 * np.pi * 440 * np.arange(8000) / 16000),
             16000, subtype="FLOAT")
    result = Predictor(ROOT, "vit_block6").predict_file(path)
    assert len(result["windows"]) == 1
    assert result["windows"][0]["valid_seconds"] == .5
    assert result["windows"][0]["padded_seconds"] == 9.5
    np.testing.assert_array_equal(result["scores"], result["windows"][0]["scores"])
    assert sum(result["scores"]) == pytest.approx(1)
