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


def test_refit_outer_reproduces_nested_oof_and_predicts_every_aligned_matrix():
    from music_genre.evaluation.crossval import refit_outer
    rng = np.random.default_rng(0)
    folds = np.repeat(np.arange(5), 12)
    y = np.tile(np.arange(3), 20)
    tracks = np.array([f"t{i}" for i in range(60)])
    x = rng.normal(size=(60, 2)) + y[:, None]

    def fit_predict(x_fit, y_fit, x_eval, c):  # nearest class mean, shrunk by c: deterministic
        means = np.stack([x_fit[y_fit == k].mean(axis=0) * c for k in range(3)])
        scores = np.exp(-((x_eval[:, None] - means) ** 2).sum(axis=2))
        return scores / scores.sum(axis=1, keepdims=True)

    oof, records = nested_cv(x, y, folds, tracks, (0.5, 1.0), fit_predict)
    shifted = x + 10
    out = refit_outer(x, y, folds, {r["fold"]: r["selected_c"] for r in records}, fit_predict,
                      {"clean": x, "shifted": shifted})
    np.testing.assert_array_equal(out["clean"], oof)
    for k, record in enumerate(records):
        held = folds == k
        expected = fit_predict(x[~held], y[~held], shifted[held], record["selected_c"])
        np.testing.assert_array_equal(out["shifted"][held], expected)
