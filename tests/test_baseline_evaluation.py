import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from music_genre.evaluation.metrics import aggregate_tracks
from music_genre.models.linear import LinearProbabilityModel


def test_track_average_gives_each_track_one_prediction():
    index = [{"track_id": "a", "window_id": i, "label_index": 0, "split": "validation"}
             for i in range(3)]
    index.append({"track_id": "b", "window_id": 0, "label_index": 1, "split": "validation"})
    probabilities = np.array([[0.8, 0.2], [0.4, 0.6], [0.6, 0.4], [0.1, 0.9]])
    ids, truth, means = aggregate_tracks(index, probabilities, 2)
    assert ids == ["a", "b"]
    np.testing.assert_array_equal(truth, [0, 1])
    np.testing.assert_allclose(means, [[0.6, 0.4], [0.1, 0.9]])


def test_duplicate_windows_and_mixed_splits_fail():
    row = {"track_id": "a", "window_id": 0, "label_index": 0, "split": "train"}
    probabilities = [[0.5, 0.5], [0.5, 0.5]]
    with pytest.raises(ValueError, match="Repeated"):
        aggregate_tracks([row, row], probabilities, 2)
    with pytest.raises(ValueError, match="split"):
        aggregate_tracks([row, {**row, "split": "validation"}], probabilities, 2)


def test_numeric_export_matches_sklearn_and_scaler_ignores_validation(tmp_path):
    rng = np.random.default_rng(42)
    x = rng.normal(size=(120, 5))
    y = np.tile([0, 1, 2], 40)
    x[:, 0] += y * 2
    pipeline = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500))
    pipeline.fit(x, y)
    validation = rng.normal(50, 10, size=(8, 5))
    scale, classifier = pipeline.steps[0][1], pipeline.steps[1][1]
    model = LinearProbabilityModel(scale.mean_, scale.scale_, classifier.coef_,
                                   classifier.intercept_, classifier.classes_)
    path = tmp_path / "model.npz"
    model.save(path)
    restored = LinearProbabilityModel.load(path)
    np.testing.assert_allclose(restored.predict_proba(validation),
                               pipeline.predict_proba(validation), atol=1e-12)
    np.testing.assert_allclose(restored.mean, x.mean(axis=0))
    assert not np.allclose(restored.mean, np.concatenate([x, validation]).mean(axis=0))
