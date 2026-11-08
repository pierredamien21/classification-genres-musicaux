import importlib.util
from pathlib import Path

import pytest


def checker():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_final_validation.py"
    spec = importlib.util.spec_from_file_location("final_validation_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.recount_prediction


def example():
    return {"schema_version": 1, "mode": "fma", "model_id": "fixed",
            "labels": ["a", "b"], "scores": [0.6, 0.4],
            "windows": [{"start_seconds": t, "valid_seconds": 10,
                         "padded_seconds": 0, "scores": p}
                        for t, p in zip([0, 10, 20], [[0.9, 0.1], [0.6, 0.4], [0.3, 0.7]])]}


def test_independent_recount_uses_all_three_windows():
    assert checker()(example(), ["a", "b"], "fixed") == pytest.approx([0.6, 0.4])


@pytest.mark.parametrize("fault", ["mean", "identity", "nonfinite", "geometry"])
def test_independent_recount_refuses_inconsistent_outputs(fault):
    item = example()
    if fault == "mean":
        item["scores"] = [0.9, 0.1]
    elif fault == "identity":
        item["labels"] = ["b", "a"]
    elif fault == "nonfinite":
        item["windows"][0]["scores"][0] = float("nan")
    else:
        item["windows"][1]["start_seconds"] = 11
    with pytest.raises(ValueError):
        checker()(item, ["a", "b"], "fixed")
