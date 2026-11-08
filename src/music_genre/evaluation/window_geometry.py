"""Compare serialized window boundaries at exact audio-sample resolution."""

import math


def window_sample_geometry(windows, sample_rate):
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("Positive integer sample rate required")
    result = []
    for window in windows:
        row = []
        for field in ("start_seconds", "valid_seconds", "padded_seconds"):
            samples = float(window[field]) * sample_rate
            # CSV stores padding as duration - valid_seconds; inference divides
            # integer padding by sample rate. Remove only floating-point roundoff.
            if not math.isfinite(samples) or samples < 0 or abs(samples - round(samples)) > 1e-7:
                raise ValueError("Window boundary is not a nonnegative sample count")
            row.append(round(samples))
        result.append(tuple(row))
    return result
