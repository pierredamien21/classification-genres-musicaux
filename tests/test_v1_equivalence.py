import importlib.util
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("verify_v1_equivalence",
                                              ROOT / "scripts/verify_v1_equivalence.py")
harness = importlib.util.module_from_spec(spec)
spec.loader.exec_module(harness)

PREDICTION = {"schema_version": 1, "model_id": "a" * 64, "mode": "fma", "labels": ["x", "y", "z"],
              "scores": [.2, .5, .3],
              "windows": [{"start_seconds": 0., "valid_seconds": 10., "padded_seconds": 0.,
                           "scores": [.2, .5, .3]}]}


def test_identical_prediction_has_zero_error():
    assert harness.compare(PREDICTION, deepcopy(PREDICTION)) == (0.0, None)


def test_harness_detects_probability_identity_geometry_and_class_changes():
    small = deepcopy(PREDICTION)
    small["windows"][0]["scores"][0] += 2e-6
    error, reason = harness.compare(PREDICTION, small)
    assert error > harness.TOLERANCE and reason is None
    changed = deepcopy(PREDICTION)
    changed["model_id"] = "b" * 64
    assert harness.compare(PREDICTION, changed) == (None, "model_id differs")
    moved = deepcopy(PREDICTION)
    moved["windows"][0]["padded_seconds"] = .5
    assert harness.compare(PREDICTION, moved) == (None, "window geometry differs")
    flipped = deepcopy(PREDICTION)
    flipped["scores"] = [.2, .3, .5]
    assert harness.compare(PREDICTION, flipped)[1] == "predicted class differs"
