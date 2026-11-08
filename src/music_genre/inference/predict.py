"""Frozen candidate inference; no labels, fitting, or dataset split decisions."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf

from music_genre.audio.processing import load_audio, segment_audio
from music_genre.data.audit import sha256_file
from music_genre.features.hpss_block6 import MusicalHpssBlock6
from music_genre.models.linear import LinearProbabilityModel


def validate_audio_file(path, *, max_file_bytes=50_000_000, max_seconds=300,
                        max_pcm_bytes=128 * 1024 * 1024):
    """Inspect the real decoder header and resource bounds before reading PCM."""
    path = Path(path)
    try:
        if path.suffix.lower() not in {".wav", ".mp3"}:
            raise ValueError("Expected WAV or MP3 audio")
        if not path.is_file() or path.stat().st_size > max_file_bytes:
            raise ValueError("Audio file size exceeds limit or file is missing")
        with sf.SoundFile(path) as source:
            if source.frames <= 0 or not 1 <= source.channels <= 8 or source.samplerate <= 0:
                raise ValueError("Audio is empty or has unsupported channels/sample rate")
            if source.frames / source.samplerate > max_seconds:
                raise ValueError("Audio duration exceeds limit")
            # load_audio requests one extra frame to detect a duration overrun.
            if (source.frames + 1) * source.channels * 4 > max_pcm_bytes:
                raise ValueError("Decoded PCM exceeds memory limit")
    except (sf.LibsndfileError, OSError) as error:
        raise ValueError(f"Cannot inspect audio: {path.name}") from error


def _inside(root, relative, area):
    path = (root / relative).resolve()
    if not path.is_relative_to((root / area).resolve()):
        raise ValueError(f"Artifact outside {area}")
    return path


def _verified_json(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f"Artifact changed: {path.name}")
    return json.loads(path.read_text(encoding="utf-8"))


class CandidatePredictor:
    """Load the prepared block-six HPSS export once, then predict unlabelled audio.

    The root must contain the pinned run, its feature report, and pretrained weights.
    Feature matrices and validation predictions are not needed by this API.
    """

    def __init__(self, root, preparation="reports/PREPARATION_EVALUATION_FINALE.json"):
        root = Path(root).resolve()
        prepared = json.loads(_inside(root, preparation, "reports").read_text(encoding="utf-8"))
        pin = prepared["runs"]["vit_block6_hpss"]
        folder = _inside(root, pin["run"]["folder"], "runs")
        report = _verified_json(folder / "report.json", pin["run"]["report_sha256"])
        cache = _inside(root, report["feature_cache"], "data/features")
        features = _verified_json(cache / "report.json", report["feature_report_sha256"])
        self.config = features["provenance"]["config"]
        self.labels = list(report["labels"])
        model_path = _inside(root, pin["model_path"], "runs")
        if (model_path != folder / "model.npz"
                or pin["model_sha256"] != report["model_sha256"]
                or sha256_file(model_path) != report["model_sha256"]
                or self.labels != prepared["labels"] or len(self.labels) != 8
                or len(set(self.labels)) != 8):
            raise ValueError("Inconsistent candidate export")
        self.model = LinearProbabilityModel.load(model_path)
        if not np.array_equal(self.model.classes, np.arange(8)) or len(self.model.mean) != 192:
            raise ValueError("Unexpected candidate dimensions or class order")
        identity = {"model_sha256": report["model_sha256"], "labels": self.labels,
                    "representation": self.config}
        self.model_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.extractor = MusicalHpssBlock6(self.config, root)

    def predict_file(self, path, *, mode="fma"):
        """Return JSON-compatible track/window probabilities in the fixed FMA mode."""
        started = perf_counter()
        if mode != "fma":
            raise ValueError("Only explicit fma window mode is supported")
        validate_audio_file(path)
        audio = load_audio(Path(path), self.config["sample_rate"])
        if not np.any(audio.samples):
            raise ValueError("Audio is digitally silent after mono conversion")
        windows = segment_audio(audio, tuple(self.config["offsets_seconds"]),
                                self.config["window_seconds"])
        features = np.stack([self.extractor.extract(window) for window in windows])
        probabilities = self.model.predict_proba(features)
        scores = probabilities.mean(axis=0)
        top = np.argsort(-scores, kind="stable")[:3]
        return {"schema_version": 1, "model_id": self.model_id, "mode": mode,
                "labels": list(self.labels), "scores": scores.tolist(),
                "top3": [{"label": self.labels[i], "score": float(scores[i])} for i in top],
                "windows": [{"start_seconds": w.start_seconds,
                             "valid_seconds": w.valid_samples / w.sample_rate,
                             "padded_seconds": (len(w.samples) - w.valid_samples) / w.sample_rate,
                             "scores": p.tolist()} for w, p in zip(windows, probabilities, strict=True)],
                "source_sample_rate": audio.source_sample_rate, "source_channels": audio.source_channels,
                "duration_seconds": len(audio.samples) / audio.sample_rate,
                "elapsed_seconds": perf_counter() - started}
