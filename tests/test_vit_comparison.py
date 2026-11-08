import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/compare_vit_baseline.py"
SPEC = importlib.util.spec_from_file_location("vit_comparison", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_comparison_rejects_protocol_drift():
    descriptor = {
        "baseline_manifest_sha256": "manifest", "labels": ["A", "B"],
        "window_counts": {"train": 6, "validation": 3},
        "track_counts": {"train": 2, "validation": 1},
        "config": {"name": "descriptors", "c_grid": [0.1, 1, 10]},
        "test_loaded": False, "test_evaluated": False,
    }
    vit = deepcopy(descriptor)
    vit["config"]["name"] = "vit"
    features = {"index_sha256": "same-windows"}
    predictions = [{"track_id": "000001", "true_label": "A"}]
    MODULE.validate_comparable(descriptor, vit, features, features, predictions, predictions)
    with pytest.raises(ValueError, match="different windows"):
        MODULE.validate_comparable(descriptor, vit, features, {"index_sha256": "changed"},
                                   predictions, predictions)
    changed = deepcopy(vit)
    changed["config"]["c_grid"] = [1]
    with pytest.raises(ValueError, match="protocols differ"):
        MODULE.validate_comparable(descriptor, changed, features, features, predictions, predictions)
    with pytest.raises(ValueError, match="not aligned"):
        MODULE.validate_comparable(descriptor, vit, features, features, predictions,
                                   [{"track_id": "000002", "true_label": "A"}])
    changed = deepcopy(vit)
    changed["test_evaluated"] = True
    with pytest.raises(ValueError, match="restricted to validation"):
        MODULE.validate_comparable(descriptor, changed, features, features, predictions, predictions)
