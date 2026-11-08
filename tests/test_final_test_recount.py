import importlib.util
from pathlib import Path

import pytest


def checker():
    path = Path(__file__).resolve().parents[1] / "scripts/verify_final_test.py"
    spec = importlib.util.spec_from_file_location("final_test_check", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.manual_metrics


def test_manual_metrics_preserve_absent_classes_and_orientation():
    result = checker()([0, 0, 1], [0, 1, 1], ["a", "b", "absent"])
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 0, 0]]
    assert result["macro_f1"] == pytest.approx(4 / 9)
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["per_class"]["a"] == {"precision": 1., "recall": .5, "f1-score": 2 / 3, "support": 2}


@pytest.mark.parametrize("truth,prediction", [([], []), ([0], [0, 1]), ([0], [2])])
def test_manual_metrics_refuse_invalid_labels_or_alignment(truth, prediction):
    with pytest.raises(ValueError):
        checker()(truth, prediction, ["a", "b"])
