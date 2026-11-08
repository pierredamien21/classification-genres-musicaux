"""Frozen inference for the nine pinned V1 systems; no labels, fitting, or split decisions."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
from threadpoolctl import threadpool_limits

from music_genre.audio.processing import load_audio, segment_audio
from music_genre.data.audit import sha256_file

PREPARATION = "reports/PREPARATION_EVALUATION_FINALE.json"
CANDIDATE = "vit_block6_hpss"
CNN_SYSTEMS = ("cnn20_seed42", "cnn20_seed43", "cnn20_seed44")
SYSTEMS = ("descriptor", "vit", "fusion", "vit_block6", "vit_hpss", CANDIDATE, *CNN_SYSTEMS)
CNN_ARCHITECTURE = {"input_shape": [1, 128, 622], "channels": [16, 32, 64], "kernel_size": 3,
                    "strides": [2, 1, 1], "padding": 1,
                    "pooling": "max2_each_block_then_global_average",
                    "memory_format": "channels_last"}


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


def _digest(identity):
    return hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()


class _NativeLogMel:
    def __init__(self, config):
        self.config = config

    def extract(self, window):
        from music_genre.features.logmel import log_mel

        if (window.sample_rate != self.config["sample_rate"]
                or len(window.samples) != round(window.sample_rate * self.config["window_seconds"])):
            raise ValueError("Unexpected CNN window geometry")
        with threadpool_limits(limits=1):
            return log_mel(window, self.config["logmel"])


class _CnnProbabilities:
    def __init__(self, model, config):
        self.native = model
        self.config = config

    def predict_proba(self, matrix):
        from music_genre.training.cnn import predict_windows

        with threadpool_limits(limits=self.config["threads"]):
            probabilities = predict_windows(self.native, matrix, np.arange(len(matrix)), self.config)
        # Historical aggregate_tracks casts windows to float64 before averaging.
        return probabilities.astype(np.float64)


class Predictor:
    """Load one prepared V1 export once, then predict unlabelled audio files.

    The root must contain the pinned run, its feature report, and pretrained weights.
    Feature matrices and validation predictions are not needed by this API.
    """

    def __init__(self, root, system=CANDIDATE, preparation=PREPARATION):
        if system not in SYSTEMS:
            raise ValueError(f"Unknown system: {system}")
        root = Path(root).resolve()
        prepared = json.loads(_inside(root, preparation, "reports").read_text(encoding="utf-8"))
        pin = prepared["runs"][system]
        folder = _inside(root, pin["run"]["folder"], "runs")
        run = _verified_json(folder / "report.json", pin["run"]["report_sha256"])
        cache = _inside(root, pin["feature_cache"], "data/features")
        features = _verified_json(cache / "report.json", pin["feature_report_sha256"])
        self.system = system
        self.labels = list(run["labels"])
        model_path = _inside(root, pin["model_path"], "runs")
        expected_name = "model.safetensors" if system in CNN_SYSTEMS else "model.npz"
        if (run["feature_cache"] != pin["feature_cache"]
                or run["feature_report_sha256"] != pin["feature_report_sha256"]
                or model_path != folder / expected_name
                or pin["model_sha256"] != run["model_sha256"]
                or sha256_file(model_path) != pin["model_sha256"]
                or self.labels != prepared["labels"] or len(self.labels) != 8
                or len(set(self.labels)) != 8):
            raise ValueError("Inconsistent pinned export")
        if system in CNN_SYSTEMS:
            self._load_cnn(run, features, pin, model_path)
        else:
            self._load_linear(root, features, pin, model_path)

    def _load_linear(self, root, features, pin, model_path):
        from music_genre.features.descriptors import MusicalDescriptors
        from music_genre.features.fusion import AlignedFusion
        from music_genre.features.vit import MusicalViT
        from music_genre.models.linear import LinearProbabilityModel

        config = features["provenance"]["config"]
        if features["provenance"]["labels"] != self.labels:
            raise ValueError("Feature labels differ from the export")
        self.model = LinearProbabilityModel.load(model_path)
        if (not np.array_equal(self.model.classes, np.arange(8))
                or len(self.model.mean) != len(features["feature_names"])):
            raise ValueError("Unexpected feature dimensions or class order")
        if self.system == "descriptor":
            self.extractor = MusicalDescriptors(config)
        elif self.system == "fusion":
            self.extractor = AlignedFusion(config, root)
        else:
            self.extractor = MusicalViT(config, root)
        # The V1 block-six cache names its columns "vit_block6_patch_mean_*"; V1 never
        # compared names for the two block-six systems, so only families are checked there.
        names_pinned = self.system in ("descriptor", "vit", "fusion", "vit_hpss")
        if ((names_pinned and self.extractor.names != features["feature_names"])
                or self.extractor.families != features["feature_families"]):
            raise ValueError("Feature order or families differ from trained export")
        self.config = config["vit"] if self.system == "fusion" else config
        self.model_id = _digest({"model_sha256": pin["model_sha256"], "labels": self.labels,
                                 "representation": config})

    def _load_cnn(self, run, features, pin, model_path):
        import torch
        from safetensors.torch import load_file

        from music_genre.models.cnn import SmallAudioCNN

        self.config = run["config"]
        self.selected_epoch = run["selected_epoch"]
        if (self.config["seed"] != int(self.system.removeprefix("cnn20_seed"))
                or self.selected_epoch != pin["selected_epoch"]
                or any(self.config[k] != v for k, v in features["provenance"]["config"].items())
                or any(self.config[k] != v for k, v in CNN_ARCHITECTURE.items())):
            raise ValueError("Inconsistent CNN export or unsupported architecture")
        native = SmallAudioCNN(len(self.labels)).to(memory_format=torch.channels_last)
        native.load_state_dict(load_file(str(model_path)), strict=True)
        native.eval().requires_grad_(False)
        self.model = _CnnProbabilities(native, self.config)
        self.extractor = _NativeLogMel(self.config)
        self.model_id = _digest({"model_sha256": pin["model_sha256"], "labels": self.labels,
                                 "config": self.config, "selected_epoch": self.selected_epoch})

    def predict_file(self, path, *, mode="fma"):
        """Return JSON-compatible track/window probabilities in the fixed FMA mode."""
        if mode != "fma":
            raise ValueError("Only explicit fma window mode is supported")
        if self.system not in CNN_SYSTEMS:
            # Same BLAS/OpenMP setting as the extraction workers that built the caches.
            with threadpool_limits(limits=1):
                return self._predict(path, self.config["offsets_seconds"], mode)
        import torch

        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(self.config["threads"])
            return self._predict(path, self.config["offsets_seconds"], mode)
        finally:
            torch.set_num_threads(previous)

    def _predict(self, path, starts, mode, extra=None):
        started = perf_counter()
        validate_audio_file(path)
        audio = load_audio(Path(path), self.config["sample_rate"])
        if not np.any(audio.samples):
            raise ValueError("Audio is digitally silent after mono conversion")
        duration = len(audio.samples) / audio.sample_rate
        windows = segment_audio(audio, tuple(starts(duration) if callable(starts) else starts),
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
                "source_sample_rate": audio.source_sample_rate,
                "source_channels": audio.source_channels, "duration_seconds": duration,
                "elapsed_seconds": perf_counter() - started, **(extra or {})}


def load_predictor(root, system):
    """Return the pinned predictor of one of the nine prepared V1 systems."""
    return Predictor(root, system)
