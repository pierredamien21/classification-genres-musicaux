from copy import deepcopy

import numpy as np
import pytest

from music_genre.evaluation.errors import align_validation, paired_outcomes, select_review


def data():
    sources = [{"track_id": "a", "genre": "Pop", "split": "validation", "artist_id": "1"},
               {"track_id": "b", "genre": "Rock", "split": "validation", "artist_id": "2"}]
    rows = [{"track_id": "b", "true_label": "Rock", "predicted_label": "Rock", "p_Pop": 0.1, "p_Rock": 0.9},
            {"track_id": "a", "true_label": "Pop", "predicted_label": "Rock", "p_Pop": 0.3, "p_Rock": 0.7}]
    return sources, rows


def test_alignment_uses_ids_not_csv_order():
    sources, rows = data()
    ids, truth, matrices = align_validation(sources, {"model": rows}, ["Pop", "Rock"])
    assert ids == ["a", "b"]
    np.testing.assert_array_equal(truth, [0, 1])
    np.testing.assert_allclose(matrices["model"], [[0.3, 0.7], [0.1, 0.9]])


@pytest.mark.parametrize("kind", ["test", "duplicate", "missing", "truth", "argmax", "nan", "sum"])
def test_invalid_error_inputs_rejected(kind):
    sources, rows = data()
    if kind == "test":
        sources[0]["split"] = "test"
    elif kind == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif kind == "missing":
        rows.pop()
    elif kind == "truth":
        rows[0]["true_label"] = "Pop"
    elif kind == "argmax":
        rows[0]["predicted_label"] = "Pop"
    else:
        rows[0]["p_Pop"] = float("nan") if kind == "nan" else 0.8
    with pytest.raises(ValueError):
        align_validation(sources, {"model": rows}, ["Pop", "Rock"])


def test_paired_corrections_are_not_confused_with_all_disagreements():
    result = paired_outcomes([0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 2])
    assert result == {"tracks": 4, "both_correct": 1, "corrected": 1,
                      "regressed": 1, "both_wrong": 1, "disagreements": 3}


def test_review_is_deterministic_artist_diverse_and_reports_shortfall():
    sources = {str(i): {"split": "validation", "artist_id": str(i // 2)} for i in range(6)}
    strata = {"rare": ["0"], "common": list(sources)}
    selected, coverage = select_review(strata, sources, 3, 42)
    assert (selected, coverage) == select_review({"common": list(reversed(list(sources))), "rare": ["0"]}, sources, 3, 42)
    artists = [sources[r["track_id"]]["artist_id"] for r in selected]
    assert len(artists) == len(set(artists)) == 3
    assert selected[0] == {"stratum": "rare", "track_id": "0"}
    assert coverage["rare"]["shortfall"] == 2
    assert coverage["common"]["shortfall"] == 1
