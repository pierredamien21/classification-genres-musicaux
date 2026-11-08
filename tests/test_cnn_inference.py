import csv
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

from music_genre.data.audit import sha256_file
from music_genre.inference.cnn_adapter import CnnPredictor

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("seed,epoch", [(42, 15), (43, 10), (44, 8)])
def test_cnn_reproduces_saved_track_with_selected_checkpoint(seed, epoch):
    source = ROOT / "reports/PREPARATION_EVALUATION_FINALE.json"
    if not source.exists():
        pytest.skip("Local prepared export required")
    prepared = json.loads(source.read_text("utf-8"))
    with (ROOT / prepared["sample"]["path"]).open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream))
    pin = prepared["runs"][f"cnn20_seed{seed}"]
    with (ROOT / pin["run"]["folder"] / "validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
        expected = next(r for r in csv.DictReader(stream) if r["track_id"] == row["track_id"])
    predictor = CnnPredictor(ROOT, seed)
    assert predictor.selected_epoch == epoch
    result = predictor.predict_file(ROOT / "data/raw" / row["audio_path"])
    assert result["labels"] == prepared["labels"]
    np.testing.assert_allclose(result["scores"], [float(expected[f"p_{label}"]) for label in prepared["labels"]],
                               atol=1e-6, rtol=0)


@pytest.mark.parametrize("seed", [42, 43, 44])
def test_short_stereo_file_has_one_vote_and_restores_threads(tmp_path, seed):
    if not (ROOT / "reports/PREPARATION_EVALUATION_FINALE.json").exists():
        pytest.skip("Local export required")
    predictor = CnnPredictor(ROOT, seed)
    before = torch.get_num_threads()
    path = tmp_path / "short.wav"
    tone = .1 * np.sin(2 * np.pi * 440 * np.arange(4000) / 8000)
    sf.write(path, np.column_stack((tone, tone)), 8000, subtype="FLOAT")
    result = predictor.predict_file(path)
    assert len(result["windows"]) == 1
    assert result["windows"][0]["valid_seconds"] == .5
    assert result["windows"][0]["padded_seconds"] == 9.5
    np.testing.assert_array_equal(result["scores"], result["windows"][0]["scores"])
    assert sum(result["scores"]) == pytest.approx(1, abs=1e-6)
    assert torch.get_num_threads() == before
    sf.write(path, np.zeros(8000), 8000)
    with pytest.raises(ValueError, match="silent"):
        predictor.predict_file(path)
    assert torch.get_num_threads() == before


@pytest.mark.parametrize("kind", ["changed_model", "wrong_epoch", "architecture", "escaped_run"])
def test_cnn_rejects_inconsistent_pinned_artifacts(tmp_path, kind):
    source = ROOT / "reports/PREPARATION_EVALUATION_FINALE.json"
    if not source.exists():
        pytest.skip("Local export metadata required")
    prepared = json.loads(source.read_text("utf-8"))
    pin = prepared["runs"]["cnn20_seed42"]
    for relative in (pin["model_path"], pin["run"]["folder"] + "/report.json",
                     pin["feature_cache"] + "/report.json"):
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    if kind == "changed_model":
        (tmp_path / pin["model_path"]).write_bytes(b"changed")
    elif kind == "escaped_run":
        pin["run"]["folder"] = "../outside"
    else:
        path = tmp_path / pin["run"]["folder"] / "report.json"
        report = json.loads(path.read_text("utf-8"))
        if kind == "wrong_epoch":
            report["selected_epoch"] = 20
        else:
            report["config"]["input_shape"] = [3, 224, 224]
        path.write_text(json.dumps(report), encoding="utf-8")
        pin["run"]["report_sha256"] = sha256_file(path)
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/PREPARATION_EVALUATION_FINALE.json").write_text(json.dumps(prepared), encoding="utf-8")
    with pytest.raises(ValueError):
        CnnPredictor(tmp_path, 42)


def test_unknown_seed_is_rejected():
    with pytest.raises(ValueError, match="seed"):
        CnnPredictor(ROOT, 45)
