"""Track-level evaluation with explicit window alignment and class order."""

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def aggregate_tracks(index, probabilities, class_count):
    probabilities = np.asarray(probabilities, dtype=np.float64)
    if (probabilities.shape != (len(index), class_count) or not len(index)
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8)):
        raise ValueError("Invalid probability matrix")
    groups = {}
    seen = set()
    splits = {row["split"] for row in index}
    if len(splits) != 1:
        raise ValueError("Aggregate one split at a time")
    for position, row in enumerate(index):
        key = (row["track_id"], int(row["window_id"]))
        if key in seen:
            raise ValueError("Repeated track/window key")
        seen.add(key)
        groups.setdefault(row["track_id"], []).append(position)
    track_ids, labels, means = [], [], []
    for track_id, positions in sorted(groups.items()):
        truths = {int(index[p]["label_index"]) for p in positions}
        if len(truths) != 1 or not 0 <= next(iter(truths)) < class_count:
            raise ValueError("Inconsistent track labels")
        track_ids.append(track_id)
        labels.append(next(iter(truths)))
        means.append(probabilities[positions].mean(axis=0))
    return track_ids, np.array(labels, dtype=np.int64), np.stack(means)


def classification_metrics(truth, probabilities, labels):
    prediction = probabilities.argmax(axis=1)
    ids = np.arange(len(labels))
    return {
        "tracks": len(truth),
        "macro_f1": float(f1_score(truth, prediction, labels=ids, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(truth, prediction)),
        "confusion_matrix": confusion_matrix(truth, prediction, labels=ids).tolist(),
        "per_class": classification_report(
            truth, prediction, labels=ids, target_names=labels, output_dict=True, zero_division=0
        ),
    }
