"""Descriptive representation drift; does not identify causal mechanisms."""

import numpy as np


def representation_drift(reference_mel, changed_mel, reference_vector, changed_vector,
                         scale, db_min, db_max):
    a, b = np.asarray(reference_mel, dtype=np.float64), np.asarray(changed_mel, dtype=np.float64)
    x, y, scale = (np.asarray(v, dtype=np.float64) for v in (reference_vector, changed_vector, scale))
    if (a.ndim != 3 or a.shape[0] != 3 or b.shape != a.shape or not a.size
            or x.ndim != 1 or not x.size or y.shape != x.shape or scale.shape != x.shape
            or np.any(scale <= 0) or not np.isfinite([db_min, db_max]).all() or db_max <= db_min
            or any(not np.isfinite(v).all() for v in (a, b, x, y, scale))):
        raise ValueError("Invalid aligned representations or bounds")
    bounded_a = np.clip((a - db_min) / (db_max - db_min), 0, 1)
    bounded_b = np.clip((b - db_min) / (db_max - db_min), 0, 1)
    delta = np.linalg.norm(y - x)
    norm_x, norm_y = np.linalg.norm(x), np.linalg.norm(y)
    channels = []
    for i, label in enumerate(('original', 'harmonic', 'percussive')):
        channels.append({
            'channel': label, 'raw_mean_absolute_db_change': float(np.abs(b[i] - a[i]).mean()),
            'raw_mean_signed_db_change': float((b[i] - a[i]).mean()),
            'bounded_mean_absolute_change': float(np.abs(bounded_b[i] - bounded_a[i]).mean()),
            'reference_below_min_fraction': float((a[i] < db_min).mean()),
            'reference_above_max_fraction': float((a[i] > db_max).mean()),
            'changed_below_min_fraction': float((b[i] < db_min).mean()),
            'changed_above_max_fraction': float((b[i] > db_max).mean()),
        })
    return {'channels': channels, 'embedding_l2': float(delta),
            'embedding_relative_l2': float(delta / norm_x) if norm_x else None,
            'scaled_delta_l2': float(np.linalg.norm((y - x) / scale)),
            'cosine_distance': float(1 - np.clip(np.dot(x, y) / (norm_x * norm_y), -1, 1))
            if norm_x and norm_y else None}
