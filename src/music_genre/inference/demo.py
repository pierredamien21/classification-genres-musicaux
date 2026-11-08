"""User-file coverage mode, separate from the frozen FMA evaluation protocol."""

import math

from threadpoolctl import threadpool_limits

from music_genre.inference.predict import Predictor

LIMITATION = "Overview sampling is not the FMA evaluation protocol; scores uncalibrated."


def overview_starts(duration: float) -> tuple[float, ...]:
    """Cover a bounded file without adding repeated votes to a short clip."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Audio duration must be finite and positive")
    last = max(0.0, duration - 10.0)
    return (0.0,) if last == 0 else (0.0, last / 2, last)


class DemoPredictor(Predictor):
    """The V1 candidate, with an extra beginning/middle/end mode for long files."""

    def predict_file(self, path, *, mode="overview"):
        if mode == "fma":
            return super().predict_file(path, mode=mode)
        if mode != "overview":
            raise ValueError("Expected fma or overview mode")
        with threadpool_limits(limits=1):
            return self._predict(path, overview_starts, mode, {"limitation": LIMITATION})
