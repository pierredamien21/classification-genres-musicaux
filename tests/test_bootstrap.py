import numpy as np
import pytest

from music_genre.evaluation.bootstrap import paired_artist_bootstrap


def example():
    tracks = [{"track_id": "t1", "artist_id": "a", "label_index": 0},
              {"track_id": "t2", "artist_id": "a", "label_index": 1},
              {"track_id": "t3", "artist_id": "b", "label_index": 1}]
    predictions = {"model": [{"track_id": "t1", "scores": [1., 0.]},
                             {"track_id": "t2", "scores": [1., 0.]},
                             {"track_id": "t3", "scores": [0., 1.]}]}
    return tracks, predictions


def test_artist_multiplicity_preserves_unequal_groups_and_fixed_labels():
    tracks, predictions = example()
    result = paired_artist_bootstrap(tracks, predictions, ["zero", "one"],
                                     artist_draws=[["a", "a"], ["b", "b"], ["a", "b"]])
    # a contains two tracks, b only one. Missing class in the second draw stays.
    np.testing.assert_allclose(result["replicate_scores"][:, 0],
                               [[1 / 3, .5], [.5, 1.], [2 / 3, 2 / 3]])
    np.testing.assert_array_equal(result["replicate_track_counts"], [4, 2, 3])
    assert result["missing_class_replicates"] == 1
    assert result["estimates"]["model"]["macro_f1"]["point"] == 2 / 3


@pytest.mark.parametrize("kind", ["missing_track", "duplicate_prediction", "extra_track",
                                  "missing_artist", "duplicate_track", "invalid_truth",
                                  "nonfinite", "bad_sum", "bad_shape", "negative"])
def test_invalid_alignment_or_metadata_is_refused(kind):
    tracks, predictions = example()
    if kind == "missing_track":
        predictions["model"].pop()
    elif kind == "duplicate_prediction":
        predictions["model"].append(predictions["model"][0])
    elif kind == "extra_track":
        predictions["model"].append({"track_id": "other", "scores": [1., 0.]})
    elif kind == "missing_artist":
        tracks[0]["artist_id"] = ""
    elif kind == "duplicate_track":
        tracks.append(tracks[0])
    elif kind == "invalid_truth":
        tracks[0]["label_index"] = -1
    else:
        predictions["model"][0]["scores"] = {
            "nonfinite": [np.nan, 0.], "bad_sum": [.2, .3],
            "bad_shape": [1.], "negative": [-1., 2.]}[kind]
    with pytest.raises(ValueError):
        paired_artist_bootstrap(tracks, predictions, ["zero", "one"], replicates=3)


def test_paired_differences_and_cnn_average_use_metrics_not_ensembled_probabilities():
    tracks, predictions = example()
    predictions["perfect"] = [{"track_id": r["track_id"], "scores": np.eye(2)[r["label_index"]]}
                              for r in tracks]
    predictions["zero"] = [{"track_id": r["track_id"], "scores": [1., 0.]} for r in tracks]
    result = paired_artist_bootstrap(
        tracks, predictions, ["zero", "one"],
        artist_draws=[["a", "a"], ["b", "b"], ["a", "b"]],
        comparisons=[("model", "model"), ("perfect", "model")],
        metric_groups={"cnn_mean": ["model", "perfect", "zero"]})
    same = result["differences"]["model - model"]
    assert same["macro_f1"] == {"point": 0., "low": 0., "high": 0.}
    paired = result["differences"]["perfect - model"]["macro_f1"]
    assert paired["point"] == pytest.approx(1 / 3)
    assert paired["low"] == pytest.approx(1 / 60)
    assert paired["high"] == pytest.approx(.65)
    assert result["estimates"]["cnn_mean"]["macro_f1"]["point"] == pytest.approx(23 / 36)
    assert result["estimates"]["cnn_mean"]["macro_f1"]["point"] != pytest.approx(2 / 3)


def test_seed_and_row_permutation_preserve_paired_results():
    tracks, predictions = example()
    first = paired_artist_bootstrap(tracks, predictions, ["zero", "one"], replicates=50)
    second = paired_artist_bootstrap(tracks[::-1], {"model": predictions["model"][::-1]},
                                      ["zero", "one"], replicates=50)
    np.testing.assert_array_equal(first["artist_draw_counts"], second["artist_draw_counts"])
    np.testing.assert_array_equal(first["replicate_scores"], second["replicate_scores"])
    assert first["estimates"] == second["estimates"]
    changed_seed = paired_artist_bootstrap(tracks, predictions, ["zero", "one"], replicates=50, seed=42)
    assert not np.array_equal(first["artist_draw_counts"], changed_seed["artist_draw_counts"])


def test_absent_classes_still_count_in_eight_label_macro_f1():
    tracks, predictions = example()
    for row in predictions["model"]:
        row["scores"] += [0.] * 6
    result = paired_artist_bootstrap(tracks, predictions, [str(i) for i in range(8)],
                                     artist_draws=[["a", "a"], ["b", "b"], ["a", "b"]])
    np.testing.assert_allclose(result["replicate_scores"][:, 0, 0], [1 / 12, 1 / 8, 1 / 6])
    assert result["missing_class_replicates"] == 3


@pytest.mark.parametrize("kwargs", [
    {"replicates": 0}, {"seed": -1}, {"artist_draws": []},
    {"artist_draws": [["a"]]}, {"artist_draws": [["a", "unknown"]]},
    {"metric_groups": {"group": ["unknown"]}},
    {"metric_groups": {"group": ["model", "model"]}},
    {"comparisons": [("model", "unknown")]},
])
def test_invalid_sampling_or_comparison_settings_are_refused(kwargs):
    tracks, predictions = example()
    with pytest.raises(ValueError):
        paired_artist_bootstrap(tracks, predictions, ["zero", "one"], **kwargs)


def test_float32_probability_roundoff_is_accepted_without_changing_predictions():
    tracks, predictions = example()
    predictions["model"][0]["scores"] = [.70000005, .30000005]
    result = paired_artist_bootstrap(tracks, predictions, ["zero", "one"],
                                     artist_draws=[["a", "b"]])
    assert result["estimates"]["model"]["macro_f1"]["point"] == 2 / 3
    predictions["model"][0]["scores"] = [.70001, .30001]
    with pytest.raises(ValueError):
        paired_artist_bootstrap(tracks, predictions, ["zero", "one"], replicates=1)
