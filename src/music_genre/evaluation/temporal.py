"""Descriptive validation diagnostics; no window or aggregation is selected."""

import numpy as np

from music_genre.evaluation.metrics import aggregate_tracks, classification_metrics


def _counts(records):
    table = [[0, 0] for _ in range(4)]
    for r in records:
        table[r["correct_windows"]][int(r["aggregate_correct"])] += 1
    return {
        "tracks": len(records), "artists": len({r["artist_id"] for r in records}),
        "disagreements": sum(r["disagreement"] for r in records),
        "aggregate_correct": sum(r["aggregate_correct"] for r in records),
        "aggregate_wrong": sum(not r["aggregate_correct"] for r in records),
        "wrong_with_correct_window": sum(not r["aggregate_correct"] and r["correct_windows"] > 0 for r in records),
        "correct_with_no_correct_window": table[0][1],
        "unique_majority_differs_from_mean": sum(
            r["majority_label"] is not None and r["majority_label"] != r["aggregate_label"] for r in records),
        "no_unique_majority": sum(r["majority_label"] is None for r in records),
        "correct_window_count_vs_mean": table,
    }


def analyze_temporal(index, probabilities, labels):
    """Return per-track and per-genre counts for three aligned validation windows.

    The 4x2 table has correct-window counts 0..3 as rows, and wrong/correct
    mean prediction as columns. A majority requires two matching argmaxes;
    three different labels have no majority. Argmax ties use class order.
    """
    if not index or any(row["split"] != "validation" for row in index):
        raise ValueError("Temporal analysis requires validation only")
    if len(labels) < 2 or len(set(labels)) != len(labels):
        raise ValueError("Distinct class labels required")
    p = np.asarray(probabilities, dtype=np.float64)
    ids, truth, means = aggregate_tracks(index, p, len(labels))
    groups = {}
    for i, row in enumerate(index):
        groups.setdefault(row["track_id"], []).append(i)
    ordered = [sorted(groups[key], key=lambda i: int(index[i]["window_id"])) for key in ids]
    for positions in ordered:
        rows = [index[i] for i in positions]
        if ([int(r["window_id"]) for r in rows] != [0, 1, 2]
                or [float(r["start_seconds"]) for r in rows] != [0., 10., 20.]):
            raise ValueError("Each track requires the three reference windows")
        for field in ("artist_id", "source_sha256"):
            if len({r[field] for r in rows}) != 1 or not rows[0][field]:
                raise ValueError("Window metadata differs")
    cube = p[ordered]
    predictions = cube.argmax(axis=2)
    aggregate = means.argmax(axis=1)
    records = []
    for t, key in enumerate(ids):
        votes = np.bincount(predictions[t], minlength=len(labels))
        majority = labels[int(votes.argmax())] if votes.max() >= 2 else None
        records.append({
            "track_id": key, "artist_id": index[ordered[t][0]]["artist_id"],
            "true_label": labels[int(truth[t])],
            "window_labels": [labels[int(x)] for x in predictions[t]],
            "window_probabilities": cube[t].tolist(), "mean_probabilities": means[t].tolist(),
            "aggregate_label": labels[int(aggregate[t])], "majority_label": majority,
            "correct_windows": int((predictions[t] == truth[t]).sum()),
            "aggregate_correct": bool(aggregate[t] == truth[t]),
            "disagreement": bool(len(set(predictions[t])) > 1),
        })
    return {"tracks": records, "all": _counts(records),
            "by_genre": {label: _counts([r for r in records if r["true_label"] == label]) for label in labels},
            "mean_metrics": classification_metrics(truth, means, labels),
            "window_metrics": [classification_metrics(truth, cube[:, w], labels) for w in range(3)]}
