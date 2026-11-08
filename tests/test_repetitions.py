from copy import deepcopy

import numpy as np
import pytest

from music_genre.evaluation.repetitions import summarize_repetitions


def reports():
    base = {"config": {"seed": 42, "max_epochs": 10}, "status": "complete",
            "test_loaded": False, "test_evaluated": False,
            "identity": {"torch_version": "same", "numpy_version": "same", "benchmark_sha256": "same",
                         "source_sha256": {"scripts/train_cnn.py": "old", "core.py": "same"}}}
    for name in ("labels", "baseline_manifest_sha256", "feature_cache", "feature_report_sha256",
                 "index_sha256", "track_counts", "window_counts", "parameters"):
        base[name] = "same"
    result = []
    for seed, score in zip([42, 43, 44], [0.3, 0.5, 0.7], strict=True):
        r = deepcopy(base)
        r["config"]["seed"] = seed
        r["validation"] = {"macro_f1": score, "accuracy": score}
        result.append(r)
    return result


def test_sample_dispersion_uses_seeds_not_windows():
    result = summarize_repetitions(reports(), [42, 43, 44])
    assert result["macro_f1"]["mean"] == pytest.approx(0.5)
    assert result["macro_f1"]["sample_std"] == pytest.approx(0.2)


@pytest.mark.parametrize("change", ["duplicate", "settings", "cache", "test", "partial", "source", "nan"])
def test_repetitions_reject_uncontrolled_comparisons(change):
    values = reports()
    item = values[1]
    if change == "duplicate":
        item["config"]["seed"] = 42
    elif change == "settings":
        item["config"]["max_epochs"] = 20
    elif change == "cache":
        item["feature_cache"] = "different"
    elif change == "test":
        item["test_evaluated"] = True
    elif change == "partial":
        item["status"] = "budget_exhausted"
    elif change == "source":
        item["identity"]["source_sha256"]["core.py"] = "changed"
    else:
        item["validation"]["macro_f1"] = np.nan
    with pytest.raises(ValueError):
        summarize_repetitions(values, [42, 43, 44])
