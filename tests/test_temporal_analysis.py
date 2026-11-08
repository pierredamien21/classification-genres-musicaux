import numpy as np
import pytest

from music_genre.evaluation.temporal import analyze_temporal


def example():
    # A: two correct windows lose to a confident error. B: the mean recovers
    # the truth even though none of the individual argmaxes predicts it.
    probabilities = np.array([
        [[.51, .49, 0], [.51, .49, 0], [.01, .99, 0]],
        [[.4, .5, .1], [.4, .1, .5], [.4, .5, .1]],
        [[.1, .2, .7]] * 3,
        [[.8, .1, .1]] * 3,
    ]).reshape(-1, 3)
    rows = [
        {"track_id": key, "window_id": str(w), "start_seconds": str(w * 10),
         "split": "validation", "label_index": str(y), "artist_id": key,
         "source_sha256": key}
        for key, y in zip("ABCD", [0, 0, 2, 1], strict=True) for w in range(3)
    ]
    return rows, probabilities


def test_temporal_report_distinguishes_mean_from_window_correctness_and_vote():
    rows, p = example()
    order = [11, 3, 8, 0, 7, 2, 6, 5, 4, 1, 10, 9]
    result = analyze_temporal([rows[i] for i in order], p[order], ["X", "Y", "Z"])
    assert result["all"] == {
        "tracks": 4, "artists": 4, "disagreements": 2, "aggregate_correct": 2,
        "aggregate_wrong": 2, "wrong_with_correct_window": 1,
        "correct_with_no_correct_window": 1, "unique_majority_differs_from_mean": 2,
        "no_unique_majority": 0,
        "correct_window_count_vs_mean": [[1, 1], [0, 0], [1, 0], [0, 1]],
    }
    assert [r["track_id"] for r in result["tracks"]] == list("ABCD")
    assert result["tracks"][0]["window_labels"] == ["X", "X", "Y"]
    assert result["tracks"][0]["aggregate_label"] == "Y"
    assert result["by_genre"]["X"]["wrong_with_correct_window"] == 1
    assert result["mean_metrics"]["accuracy"] == .5
    assert [m["accuracy"] for m in result["window_metrics"]] == [.5, .5, .25]


@pytest.mark.parametrize("kind", ["test", "train", "offset", "window_id", "artist", "source", "nan", "probability_sum"])
def test_temporal_report_refuses_incomparable_windows(kind):
    rows, p = example()
    if kind in ("test", "train"):
        for row in rows:
            row["split"] = kind
    elif kind == "offset":
        rows[0]["start_seconds"] = "5"
    elif kind == "window_id":
        rows[0]["window_id"] = "7"
    elif kind == "artist":
        rows[0]["artist_id"] = "different"
    elif kind == "source":
        rows[0]["source_sha256"] = "different"
    elif kind == "nan":
        p[0, 0] = np.nan
    else:
        p[0, 0] += .001
    with pytest.raises(ValueError):
        analyze_temporal(rows, p, ["X", "Y", "Z"])
