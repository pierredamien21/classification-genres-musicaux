"""Fixed-protocol statistics for completed campaigns; no inference or fitting."""

import numpy as np

from music_genre.evaluation.bootstrap import paired_artist_bootstrap
from music_genre.evaluation.metrics import classification_metrics

EXPECTED_BOOTSTRAP = {
    "unit": "artist", "resampling": "all_tracks_of_each_sampled_artist_with_multiplicity",
    "paired_across_systems": True, "replicates": 2000, "seed": 20261028,
    "generator": "numpy_PCG64", "confidence_level": .95, "interval": "percentile",
    "quantile_method": "linear", "resample_tracks_within_artist": False,
    "stratify_by_genre": False, "missing_class_replicates": "retain_fixed_labels_and_report_count",
    "claim": "approximate_pointwise_intervals_conditional_on_fixed_models_not_simultaneous_tests",
}


def summarize_predictions(tracks, predictions, labels, protocol, train_counts):
    """Return class metrics, paired intervals and their resampling audit arrays."""
    names = protocol["linear_systems"] + protocol["cnn_runs"]
    if (len(names) != len(set(names)) or set(predictions) != set(names)
            or len(protocol["cnn_runs"]) != 3 or "majority" in names or "cnn20_mean" in names
            or protocol["bootstrap"] != EXPECTED_BOOTSTRAP
            or protocol["aggregation"] != "mean_of_window_probabilities"
            or protocol["refit_on_train_and_validation"] is not False):
        raise ValueError("Unsupported protocol or model roster")
    counts = np.asarray(train_counts)
    if (counts.shape != (len(labels),) or counts.dtype.kind not in "iu"
            or np.any(counts < 0) or counts.sum() == 0):
        raise ValueError("Invalid train class counts")
    majority = int(counts.argmax())
    ordered = sorted(tracks, key=lambda row: row["track_id"])
    scores = {name: predictions[name] for name in names}
    scores["majority"] = [{"track_id": t["track_id"], "scores": np.eye(len(labels))[majority].tolist()}
                          for t in ordered]
    comparisons = [protocol["primary_comparison"], *protocol["secondary_comparisons"]]
    groups = {"cnn20_mean": protocol["cnn_runs"]}
    result = paired_artist_bootstrap(ordered, scores, labels, replicates=2000, seed=20261028,
                                     comparisons=comparisons, metric_groups=groups)
    arrays = {key: result.pop(key) for key in ("replicate_scores", "artist_draw_counts",
                                             "replicate_track_counts")}
    truth = np.array([r["label_index"] for r in ordered])
    metrics = {}
    for name, records in scores.items():
        by_id = {r["track_id"]: r["scores"] for r in records}
        probabilities = np.array([by_id[r["track_id"]] for r in ordered])
        metrics[name] = classification_metrics(truth, probabilities, labels)
        for metric in ("macro_f1", "accuracy"):
            if abs(metrics[name][metric] - result["estimates"][name][metric]["point"]) > 1e-12:
                raise ValueError("Bootstrap and classification point estimates disagree")
    result.update({"classification": metrics, "labels": labels, "tracks": len(ordered),
                   "artists": len(result["artist_ids"]), "settings": dict(EXPECTED_BOOTSTRAP),
                   "comparisons": comparisons, "metric_groups": groups,
                   "majority_train_class": labels[majority], "train_class_counts": counts.tolist(),
                   "cnn_point_sample_std": {metric: float(np.std([
                       metrics[n][metric] for n in protocol["cnn_runs"]], ddof=1))
                       for metric in ("macro_f1", "accuracy")},
                   "sampled_track_count_range": [int(arrays["replicate_track_counts"].min()),
                                                 int(arrays["replicate_track_counts"].max())]})
    return result, arrays
