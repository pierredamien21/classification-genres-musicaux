"""Nested, artist-grouped cross-validation on window features (V2 development).

For each outer fold, the regularization value is chosen by leave-one-fold-out over the other
folds only, then refitted on all of them and used to predict the outer fold. Scores are track
macro-F1 on the mean of window probabilities, as in V1.
"""

import numpy as np
from sklearn.metrics import f1_score


def track_mean(track_ids, labels, probabilities):
    """Mean window probabilities per track, tracks in first-seen order."""
    tracks, first, inverse = np.unique(track_ids, return_index=True, return_inverse=True)
    sums = np.zeros((len(tracks), probabilities.shape[1]))
    np.add.at(sums, inverse, probabilities)
    probs = sums / np.bincount(inverse)[:, None]
    truth = labels[first]
    if np.any(labels != truth[inverse]):
        raise ValueError("A track has windows with different labels")
    order = np.argsort(first)
    return tracks[order], truth[order], probs[order]


def macro_f1(track_ids, labels, probabilities):
    _, truth, probs = track_mean(track_ids, labels, probabilities)
    return float(f1_score(truth, probs.argmax(axis=1), average="macro", zero_division=0))


def nested_cv(x, y, folds, track_ids, c_grid, fit_predict):
    """Return out-of-fold window probabilities and one record per outer fold.

    fit_predict(x_fit, y_fit, x_eval, c) -> probabilities for x_eval. Ties pick the smaller c.
    """
    order = sorted(set(folds.tolist()))
    oof = np.full((len(x), len(set(y.tolist()))), np.nan)
    records = []
    for outer in order:
        train = folds != outer
        inner = {}
        for c in c_grid:
            inner[c] = [macro_f1(track_ids[folds == k], y[folds == k],
                                 fit_predict(x[train & (folds != k)], y[train & (folds != k)],
                                             x[folds == k], c))
                        for k in order if k != outer]
        best = max(sorted(c_grid), key=lambda c: np.mean(inner[c]))  # max keeps the first, smallest c
        held = folds == outer
        oof[held] = fit_predict(x[train], y[train], x[held], best)
        records.append({"fold": outer, "selected_c": best,
                        "inner_macro_f1": {str(c): float(np.mean(s)) for c, s in inner.items()},
                        "macro_f1": macro_f1(track_ids[held], y[held], oof[held]),
                        "tracks": len(set(track_ids[held].tolist()))})
    return oof, records
