"""Read-only checks of selected validation windows against stored model inputs."""

import numpy as np


def diagnose_window(window, recomputed, cached, index, tolerance=1e-4):
    if (index["split"] != "validation" or window.sample_rate != 16000
            or len(window.samples) != 160000 or window.samples.ndim != 1
            or not 0 < window.valid_samples <= len(window.samples)
            or not np.isfinite(window.samples).all()):
        raise ValueError("Expected a valid native validation window")
    for name, value in (("start_seconds", window.start_seconds),
                        ("valid_seconds", window.valid_samples / window.sample_rate),
                        ("padded_seconds", (len(window.samples) - window.valid_samples) / window.sample_rate)):
        if not np.isclose(float(index[name]), value, rtol=0, atol=1e-8):
            raise ValueError("Window timing differs from cached index")
    if (recomputed.shape != (128, 622) or cached.shape != recomputed.shape
            or not np.isfinite(recomputed).all() or not np.isfinite(cached).all()
            or not np.isfinite(tolerance) or tolerance < 0):
        raise ValueError("Expected finite native log-Mel matrices")
    error = float(np.max(np.abs(recomputed - cached)))
    if error > tolerance:
        raise ValueError(f"Spectrogram differs from cache: {error:g} dB")
    valid = window.samples[:window.valid_samples].astype(np.float64)
    rms = float(np.sqrt(np.mean(valid ** 2)))
    return {
        "start_seconds": window.start_seconds,
        "valid_seconds": window.valid_samples / window.sample_rate,
        "padded_seconds": (len(window.samples) - window.valid_samples) / window.sample_rate,
        "valid_rms_dbfs": float(20 * np.log10(rms)) if rms > 0 else None,
        "valid_peak_abs": float(np.max(np.abs(valid))),
        "max_cache_error_db": error,
        "below_db_range_fraction": float(np.mean(recomputed < -80)),
        "above_db_range_fraction": float(np.mean(recomputed > 20)),
    }
