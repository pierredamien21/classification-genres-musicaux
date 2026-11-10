"""V2 phase 4: temperature calibration and the "unknown genre" rejection threshold.

Both are fitted on out-of-fold development probabilities only (docs/PROTOCOLE_V2.md). The
probabilities are track means of window probabilities, so the temperature rescales their logs.
"""

import numpy as np
from scipy.optimize import minimize_scalar

EPSILON = 1e-12


def temperature_scale(probs, temperature):
    logits = np.log(np.clip(probs, EPSILON, 1)) / temperature
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def fit_temperature(probs, truth, bounds=(0.05, 20.0)):
    """Temperature minimizing the mean negative log-likelihood (searched on log T)."""
    def nll(log_t):
        scaled = temperature_scale(probs, np.exp(log_t))
        return -np.mean(np.log(np.clip(scaled[np.arange(len(truth)), truth], EPSILON, 1)))
    result = minimize_scalar(nll, bounds=np.log(bounds), method="bounded", options={"xatol": 1e-6})
    if not result.success:
        raise RuntimeError("Temperature fit failed")
    return float(np.exp(result.x))


def expected_calibration_error(probs, truth, bins=15):
    """Equal-width confidence bins on the top probability, weighted by bin size."""
    confidence, correct = probs.max(axis=1), probs.argmax(axis=1) == truth
    edges = np.linspace(0, 1, bins + 1)
    position = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    error = 0.0
    for b in range(bins):
        members = position == b
        if members.any():
            error += members.mean() * abs(confidence[members].mean() - correct[members].mean())
    return float(error)


def cross_fit_temperature(probs, truth, folds):
    """Calibrate each fold with a temperature fitted on the other folds only."""
    calibrated, temperatures = np.empty_like(probs), {}
    for k in sorted(set(folds.tolist())):
        held = folds == k
        temperatures[k] = fit_temperature(probs[~held], truth[~held])
        calibrated[held] = temperature_scale(probs[held], temperatures[k])
    return calibrated, temperatures


def rejection_threshold(confidence, rate):
    """Top-probability threshold below which `rate` of the in-domain tracks would be rejected."""
    if not 0 < rate < 1:
        raise ValueError("Rejection rate must be in (0, 1)")
    return float(np.quantile(confidence, rate))
