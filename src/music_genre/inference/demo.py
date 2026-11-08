"""User-file coverage mode, separate from the frozen FMA evaluation adapter."""

import math
from pathlib import Path
from time import perf_counter

import numpy as np

from music_genre.audio.processing import load_audio, segment_audio
from music_genre.inference.predict import CandidatePredictor, validate_audio_file


def overview_starts(duration: float) -> tuple[float, ...]:
    """Cover a bounded file without adding repeated votes to a short clip."""
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("Audio duration must be finite and positive")
    last = max(0.0, duration - 10.0)
    return (0.0,) if last == 0 else (0.0, last / 2, last)


class DemoPredictor(CandidatePredictor):
    def predict_file(self, path, *, mode="overview"):
        if mode == "fma":
            return super().predict_file(path, mode=mode)
        if mode != "overview":
            raise ValueError("Expected fma or overview mode")
        started = perf_counter()
        validate_audio_file(path)
        audio = load_audio(Path(path), self.config["sample_rate"])
        if not np.any(audio.samples):
            raise ValueError("Audio is digitally silent after mono conversion")
        duration = len(audio.samples) / audio.sample_rate
        windows = segment_audio(audio, overview_starts(duration), 10.0)
        features = np.stack([self.extractor.extract(w) for w in windows])
        probabilities = self.model.predict_proba(features)
        scores = probabilities.mean(axis=0)
        top = np.argsort(-scores, kind="stable")[:3]
        return {
            "schema_version": 1, "model_id": self.model_id, "mode": mode,
            "labels": list(self.labels), "scores": scores.tolist(),
            "top3": [{"label": self.labels[i], "score": float(scores[i])} for i in top],
            "windows": [{"start_seconds": w.start_seconds,
                         "valid_seconds": w.valid_samples / w.sample_rate,
                         "padded_seconds": (len(w.samples) - w.valid_samples) / w.sample_rate,
                         "scores": p.tolist()} for w, p in zip(windows, probabilities, strict=True)],
            "source_sample_rate": audio.source_sample_rate,
            "source_channels": audio.source_channels, "duration_seconds": duration,
            "elapsed_seconds": perf_counter() - started,
            "limitation": "Overview sampling is not the FMA evaluation protocol; scores uncalibrated.",
        }
