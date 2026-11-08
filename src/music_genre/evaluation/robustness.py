"""Predeclared, non-destructive perturbations of valid mono audio, before padding."""

import hashlib

import numpy as np

from music_genre.evaluation.metrics import classification_metrics


def perturb_audio(samples, condition, track_id, seed=20261028):
    samples = np.asarray(samples, dtype=np.float32)
    if samples.ndim != 1 or not len(samples) or not np.isfinite(samples).all():
        raise ValueError("Expected nonempty finite mono signal before padding")
    if condition == "identity":
        return samples.copy()
    if condition == "gain_minus_6_db":
        return (samples * 10 ** (-6 / 20)).astype(np.float32)
    if condition == "white_noise_snr_20_db":
        key = f"{seed}:{track_id}:{condition}".encode()
        rng = np.random.Generator(np.random.PCG64(int(hashlib.sha256(key).hexdigest()[:16], 16)))
        noise = rng.standard_normal(len(samples))
        signal_rms = np.sqrt(np.mean(samples.astype(np.float64) ** 2))
        if signal_rms == 0:
            raise ValueError("SNR is undefined for a silent signal")
        noise *= signal_rms / (10 * np.sqrt(np.mean(noise ** 2)))
        return (samples.astype(np.float64) + noise).astype(np.float32)
    raise ValueError("Unknown perturbation")


def compare_condition(truth, identity, changed, labels, thresholds):
    """Compare aligned track probabilities; threshold crossings request review only."""
    truth = np.asarray(truth)
    identity, changed = np.asarray(identity), np.asarray(changed)
    if (truth.ndim != 1 or not len(truth) or len(set(labels)) != len(labels)
            or not np.issubdtype(truth.dtype, np.integer)
            or np.any(truth < 0) or np.any(truth >= len(labels))):
        raise ValueError("Invalid truth or labels")
    for matrix in (identity, changed):
        if (matrix.shape != (len(truth), len(labels)) or not np.isfinite(matrix).all()
                or np.any(matrix < 0) or not np.allclose(matrix.sum(axis=1), 1, atol=1e-6, rtol=0)):
            raise ValueError("Invalid or unaligned probabilities")
    metrics = classification_metrics(truth, changed, labels)
    base = classification_metrics(truth, identity, labels)
    drop = base["macro_f1"] - metrics["macro_f1"]
    flipped = int((identity.argmax(axis=1) != changed.argmax(axis=1)).sum())
    fraction = flipped / len(truth)
    return {"metrics": metrics, "macro_f1_drop": drop, "accuracy_difference": metrics["accuracy"] - base["accuracy"],
            "flipped_tracks": flipped, "flip_fraction": fraction,
            "mean_absolute_probability_change": float(np.abs(changed - identity).mean()),
            "max_absolute_probability_change": float(np.abs(changed - identity).max()),
            "review_triggers": {"macro_f1_drop": drop >= thresholds["macro_f1_drop_at_least"],
                                "flip_fraction": fraction >= thresholds["prediction_flip_fraction_at_least"]}}
