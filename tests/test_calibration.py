import numpy as np
import pytest

from music_genre.evaluation.calibration import (
    cross_fit_temperature,
    expected_calibration_error,
    fit_temperature,
    rejection_threshold,
    temperature_scale,
)


def softmax(z):
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def synthetic(n=6000, seed=0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(scale=1.5, size=(n, 4))
    truth = np.array([rng.choice(4, p=p) for p in softmax(logits)])
    return logits, truth


def test_temperature_one_is_the_identity_and_large_temperatures_flatten():
    p = softmax(np.array([[2.0, 0.0, -1.0]]))
    np.testing.assert_allclose(temperature_scale(p, 1.0), p)
    assert temperature_scale(p, 5.0).max() < p.max()


def test_fit_temperature_undoes_a_known_overconfidence():
    logits, truth = synthetic()
    assert fit_temperature(softmax(2 * logits), truth) == pytest.approx(2, rel=0.1)


def test_ece_on_a_worked_example():
    probs = np.array([[0.9, 0.1], [0.9, 0.1], [0.6, 0.4], [0.6, 0.4]])
    truth = np.array([0, 1, 0, 0])
    # bin 0.9: confidence 0.9, accuracy 0.5; bin 0.6: confidence 0.6, accuracy 1.0
    assert expected_calibration_error(probs, truth, bins=10) == pytest.approx(0.5 * 0.4 + 0.5 * 0.4)


def test_cross_fitted_temperature_never_uses_the_fold_it_calibrates():
    logits, truth = synthetic(3000)
    probs = softmax(2 * logits)
    folds = np.arange(len(truth)) % 3
    calibrated, temperatures = cross_fit_temperature(probs, truth, folds)
    for k in range(3):
        held = folds == k
        t = fit_temperature(probs[~held], truth[~held])
        assert temperatures[k] == pytest.approx(t)
        np.testing.assert_allclose(calibrated[held], temperature_scale(probs[held], t))


def test_rejection_threshold_rejects_the_requested_share():
    confidence = np.linspace(0.2, 1.0, 1000)
    threshold = rejection_threshold(confidence, 0.05)
    assert np.mean(confidence < threshold) == pytest.approx(0.05, abs=0.002)
