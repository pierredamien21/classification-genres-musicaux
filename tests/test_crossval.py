import numpy as np
import pytest

from music_genre.evaluation.crossval import nested_cv, track_mean


def test_track_mean_averages_windows_in_first_seen_order():
    tracks, truth, probs = track_mean(np.array(["b", "a", "b"]), np.array([1, 0, 1]),
                                      np.array([[0.2, 0.8], [1.0, 0.0], [0.4, 0.6]]))
    assert tracks.tolist() == ["b", "a"] and truth.tolist() == [1, 0]
    assert np.allclose(probs, [[0.3, 0.7], [1.0, 0.0]])


def test_track_mean_rejects_a_track_with_two_labels():
    with pytest.raises(ValueError):
        track_mean(np.array(["a", "a"]), np.array([0, 1]), np.full((2, 2), 0.5))


def test_outer_fold_never_reaches_any_fit_and_ties_pick_the_smaller_c():
    folds = np.repeat(np.arange(5), 6)
    y = np.repeat(np.tile([0, 1, 2], 5), 2)  # one label per track
    tracks = np.array([f"t{i // 2}" for i in range(30)])  # two windows per track, same fold
    x = folds[:, None].astype(float)  # column 0 reveals the fold of every training row
    seen = []

    def fit_predict(x_fit, y_fit, x_eval, c):
        seen.append((set(x_fit[:, 0].astype(int)), set(x_eval[:, 0].astype(int)), c))
        return np.eye(3)[np.zeros(len(x_eval), dtype=int)]  # same score for every C

    oof, folds_report = nested_cv(x, y, folds, tracks, (1.0, 0.1), fit_predict)
    assert oof.shape == (30, 3) and np.isfinite(oof).all()
    for record in folds_report:
        assert record["selected_c"] == 0.1
        assert len(record["inner_macro_f1"]) == 2
    # Every fit excludes the fold it predicts, and inner fits exclude the outer fold too.
    assert all(not fit & held for fit, held, _ in seen)
    outer_calls = [(fit, held) for fit, held, _ in seen if len(fit) == 4]
    assert sorted(min(held) for _, held in outer_calls) == [0, 1, 2, 3, 4]
    assert all(len(fit) == 3 for fit, _, _ in seen if len(fit) != 4)
    assert len(seen) == 5 * (4 * 2 + 1)
