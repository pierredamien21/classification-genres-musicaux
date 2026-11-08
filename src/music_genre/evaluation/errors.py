"""Validation-only error comparisons and reproducible artist-diverse review queues."""

import hashlib

import numpy as np


def align_validation(manifest, predictions, labels):
    if not manifest or any(r["split"] != "validation" for r in manifest):
        raise ValueError("Only validation rows are allowed")
    sources = {r["track_id"]: r for r in manifest}
    if len(sources) != len(manifest):
        raise ValueError("Duplicate manifest track")
    ids = sorted(sources)
    truth = np.array([labels.index(sources[key]["genre"]) for key in ids])
    matrices = {}
    for name, rows in predictions.items():
        indexed = {r["track_id"]: r for r in rows}
        if len(indexed) != len(rows) or set(indexed) != set(ids):
            raise ValueError("Missing, extra or duplicate prediction track")
        if any(indexed[key]["true_label"] != sources[key]["genre"] for key in ids):
            raise ValueError("Prediction truth differs from manifest")
        probs = np.array([[float(indexed[key][f"p_{label}"]) for label in labels] for key in ids])
        if (not np.isfinite(probs).all() or np.any(probs < 0)
                or not np.allclose(probs.sum(axis=1), 1, atol=1e-8)):
            raise ValueError("Invalid probabilities")
        if any(indexed[key]["predicted_label"] != labels[int(pred)]
               for key, pred in zip(ids, probs.argmax(axis=1), strict=True)):
            raise ValueError("Saved prediction disagrees with argmax")
        matrices[name] = probs
    return ids, truth, matrices


def paired_outcomes(truth, reference, candidate):
    truth, reference, candidate = map(np.asarray, (truth, reference, candidate))
    if truth.ndim != 1 or reference.shape != truth.shape or candidate.shape != truth.shape:
        raise ValueError("Paired predictions must be aligned one-dimensional arrays")
    a, b = reference == truth, candidate == truth
    return {"tracks": len(truth), "both_correct": int((a & b).sum()),
            "corrected": int((~a & b).sum()), "regressed": int((a & ~b).sum()),
            "both_wrong": int((~a & ~b).sum()), "disagreements": int((reference != candidate).sum())}


def genre_changes(truth, reference, candidate, labels):
    """Separate changes in recall from false positives originating in other genres."""
    truth, reference, candidate = map(np.asarray, (truth, reference, candidate))
    paired_outcomes(truth, reference, candidate)  # Validate alignment before masking.
    result = {}
    for index, label in enumerate(labels):
        mask = truth == index
        result[label] = {
            **paired_outcomes(truth[mask], reference[mask], candidate[mask]),
            "reference_false_positives": int(((reference == index) & ~mask).sum()),
            "candidate_false_positives": int(((candidate == index) & ~mask).sum()),
        }
    return result


def select_review(strata, sources, per_stratum, seed):
    """Fill rare strata first; sample by seeded hash, at most one track per artist."""
    if per_stratum < 1:
        raise ValueError("Positive review quota required")
    selected, coverage, used_tracks, used_artists = [], {}, set(), set()
    for name in sorted(strata, key=lambda key: (len(set(strata[key])), key)):
        ids = sorted(set(strata[name]), key=lambda key: (
            hashlib.sha256(f"{seed}:{name}:{key}".encode()).hexdigest(), key))
        chosen = []
        for key in ids:
            row = sources[key]
            if row["split"] != "validation" or not row["artist_id"]:
                raise ValueError("Review requires validation and known artists")
            if key not in used_tracks and row["artist_id"] not in used_artists and len(chosen) < per_stratum:
                chosen.append(key)
                used_tracks.add(key)
                used_artists.add(row["artist_id"])
                selected.append({"stratum": name, "track_id": key})
        coverage[name] = {"available_tracks": len(ids), "selected": len(chosen),
                          "requested": per_stratum, "shortfall": per_stratum - len(chosen)}
    return selected, coverage
