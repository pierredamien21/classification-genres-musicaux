"""Paired artist-cluster percentile bootstrap of fixed track classifiers."""

import numpy as np

METRICS = ("macro_f1", "accuracy")


def _scores(confusions):
    true_counts = confusions.sum(axis=-1)
    predicted_counts = confusions.sum(axis=-2)
    diagonal = np.diagonal(confusions, axis1=-2, axis2=-1)
    denominator = true_counts + predicted_counts
    f1 = np.divide(2 * diagonal, denominator, out=np.zeros_like(diagonal, dtype=float),
                   where=denominator != 0)
    return np.stack((f1.mean(axis=-1), diagonal.sum(axis=-1) / true_counts.sum(axis=-1)), axis=-1)


def paired_artist_bootstrap(tracks, predictions, labels, *, replicates=2000,
                            seed=20261028, artist_draws=None, comparisons=(), metric_groups=None):
    """Align track scores, resample whole artists, and return scores plus audit arrays.

    tracks: records with track_id, artist_id, integer label_index.
    predictions: model -> records with track_id and scores in labels order.
    Explicit artist_draws are for reproducible worked examples; otherwise use
    PCG64. Sorted artist and track IDs make the result invariant to row order.
    Metrics remain track-weighted, with zero F1 for every undefined fixed class.
    metric_groups average already computed model metrics (not probabilities),
    using the same artist draws. comparisons are ordered (left, right) pairs.
    """
    if (not tracks or not predictions or len(labels) < 2
            or any(not isinstance(label, str) or not label.strip() for label in labels)
            or len(set(labels)) != len(labels)):
        raise ValueError("Nonempty tracks, models and distinct labels required")
    for row in tracks:
        if (any(not isinstance(row.get(key), str) or not row[key].strip()
                for key in ("track_id", "artist_id"))
                or type(row.get("label_index")) is not int
                or not 0 <= row["label_index"] < len(labels)):
            raise ValueError("Missing track/artist identity or invalid truth")
    if len({r["track_id"] for r in tracks}) != len(tracks):
        raise ValueError("Duplicate track identity")
    if type(replicates) is not int or replicates < 1 or type(seed) is not int or seed < 0:
        raise ValueError("Positive replicate count and nonnegative integer seed required")
    ordered = sorted(tracks, key=lambda row: row["track_id"])
    ids = [r["track_id"] for r in ordered]
    artists = sorted({r["artist_id"] for r in ordered})
    artist_position = {artist: i for i, artist in enumerate(artists)}
    group = np.array([artist_position[r["artist_id"]] for r in ordered])
    truth = np.array([r["label_index"] for r in ordered])
    models = list(predictions)
    class_count = len(labels)
    grouped = np.zeros((len(artists), len(models), class_count, class_count), dtype=np.int64)
    for m, name in enumerate(models):
        if not isinstance(name, str) or not name.strip():
            raise ValueError("Model name required")
        rows = {r["track_id"]: r for r in predictions[name]}
        if len(rows) != len(predictions[name]) or set(rows) != set(ids):
            raise ValueError("Missing, extra or duplicated prediction track")
        scores = np.array([rows[track]["scores"] for track in ids], dtype=float)
        if (scores.shape != (len(ids), class_count) or not np.isfinite(scores).all()
                # CNN softmax exports retain float32 roundoff; protocol tolerance.
                or np.any(scores < 0) or not np.allclose(scores.sum(axis=1), 1, atol=1e-6, rtol=0)):
            raise ValueError("Invalid probability matrix or label dimensions")
        predicted = scores.argmax(axis=1)
        np.add.at(grouped, (group, m, truth, predicted), 1)
    if artist_draws is None:
        rng = np.random.Generator(np.random.PCG64(seed))
        draws = rng.integers(0, len(artists), size=(replicates, len(artists)))
    else:
        if (not len(artist_draws) or any(len(draw) != len(artists)
                or any(a not in artist_position for a in draw) for draw in artist_draws)):
            raise ValueError("Each explicit draw must contain exactly the artist count")
        draws = np.array([[artist_position[a] for a in draw] for draw in artist_draws])
        replicates = len(draws)
    counts = np.zeros((replicates, len(artists)), dtype=np.int64)
    np.add.at(counts, (np.arange(replicates)[:, None], draws), 1)
    confusions = (counts @ grouped.reshape(len(artists), -1)).reshape(
        replicates, len(models), class_count, class_count)
    values = _scores(confusions)
    point = _scores(grouped.sum(axis=0))
    original_models = list(models)
    for name, members in (metric_groups or {}).items():
        if (not isinstance(name, str) or not name.strip() or name in models or not members
                or len(set(members)) != len(members) or any(m not in original_models for m in members)):
            raise ValueError("Invalid metric group")
        positions = [models.index(m) for m in members]
        values = np.concatenate((values, values[:, positions].mean(axis=1, keepdims=True)), axis=1)
        point = np.concatenate((point, point[positions].mean(axis=0, keepdims=True)), axis=0)
        models.append(name)
    estimates = {}
    for m, model in enumerate(models):
        estimates[model] = {}
        for k, metric in enumerate(METRICS):
            low, high = np.quantile(values[:, m, k], [.025, .975], method="linear")
            estimates[model][metric] = {"point": float(point[m, k]),
                                       "low": float(low), "high": float(high)}
    differences = {}
    for left, right in comparisons:
        if left not in models or right not in models:
            raise ValueError("Unknown comparison model")
        a, b = models.index(left), models.index(right)
        key = f"{left} - {right}"
        if key in differences:
            raise ValueError("Duplicate comparison")
        differences[key] = {}
        for k, metric in enumerate(METRICS):
            low, high = np.quantile(values[:, a, k] - values[:, b, k], [.025, .975], method="linear")
            differences[key][metric] = {"point": float(point[a, k] - point[b, k]),
                                        "low": float(low), "high": float(high)}
    truth_counts = confusions[:, 0].sum(axis=-1)
    return {"models": models, "metrics": list(METRICS), "artist_ids": artists,
            "estimates": estimates, "differences": differences,
            "replicate_scores": values, "artist_draw_counts": counts,
            "replicate_track_counts": truth_counts.sum(axis=1),
            "missing_class_replicates": int(np.any(truth_counts == 0, axis=1).sum())}
