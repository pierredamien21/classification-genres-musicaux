import json
from pathlib import Path

import pytest

from music_genre.evaluation.final_reporting import summarize_predictions


def inputs():
    protocol = json.loads((Path(__file__).resolve().parents[1] /
                           "configs/final_evaluation_protocol.json").read_text("utf-8"))
    protocol.update(linear_systems=["candidate", "reference"],
                    cnn_runs=["cnn1", "cnn2", "cnn3"],
                    primary_comparison=["candidate", "reference"], secondary_comparisons=[])
    tracks = [{"track_id": "1", "artist_id": "A", "label_index": 0},
              {"track_id": "2", "artist_id": "B", "label_index": 1}]
    scores = {"candidate": [[0.9, 0.1], [0.01, 0.99]],
              "reference": [[0.6, 0.4], [0.6, 0.4]],
              "cnn1": [[0.6, 0.4], [0.6, 0.4]],
              "cnn2": [[0.6, 0.4], [0.6, 0.4]],
              "cnn3": [[0.9, 0.1], [0.01, 0.99]]}
    predictions = {name: [{"track_id": str(i + 1), "scores": row}
                         for i, row in enumerate(values)] for name, values in scores.items()}
    return tracks, predictions, ["a", "b"], protocol, [1, 9]


def test_report_uses_train_majority_and_mean_cnn_metrics_not_ensemble():
    report, arrays = summarize_predictions(*inputs())
    assert report["majority_train_class"] == "b"
    assert report["classification"]["majority"]["confusion_matrix"] == [[0, 1], [0, 1]]
    assert report["classification"]["candidate"]["macro_f1"] == 1
    assert report["estimates"]["cnn20_mean"]["macro_f1"]["point"] == pytest.approx(5 / 9)
    assert report["differences"]["candidate - reference"]["accuracy"]["point"] == .5
    assert arrays["replicate_scores"].shape == (2000, 7, 2)


@pytest.mark.parametrize("fault", ["missing_model", "partial_cohort", "wrong_bootstrap", "bad_counts"])
def test_report_refuses_protocol_or_cohort_drift(fault):
    args = list(inputs())
    if fault == "missing_model":
        del args[1]["cnn2"]
    elif fault == "partial_cohort":
        args[1]["candidate"].pop()
    elif fault == "wrong_bootstrap":
        args[3]["bootstrap"]["unit"] = "window"
    else:
        args[4] = [0, 0]
    with pytest.raises(ValueError):
        summarize_predictions(*args)
