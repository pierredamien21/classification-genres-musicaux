"""Frozen inference for the V2 systems pinned by lock 3; no labels, fitting or decisions.

MaestPredictor: MAEST block 12 tokens and the logistic head refitted on the whole development set
(scripts/fit_final_v2.py). CnnV2Predictor: one reinforced CNN seed (scripts/train_cnn.py,
configs/cnn_v2.json). Both rebuild the windows of their development caches: same decoding,
offsets, window length and per-window feature code. The V1 systems use inference/predict.py.
"""

import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from music_genre.audio.processing import load_audio, segment_audio
from music_genre.data.audit import sha256_file


def _verified_json(path, expected):
    if sha256_file(path) != expected:
        raise ValueError(f"Pinned file changed: {path}")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _windows(path, config):
    audio = load_audio(Path(path), config["sample_rate"])
    if not np.any(audio.samples):
        raise ValueError("Audio is digitally silent after mono conversion")
    windows = segment_audio(audio, tuple(config["offsets_seconds"]), config["window_seconds"])
    if not windows:
        raise ValueError("No audio windows")
    return windows


class MaestPredictor:
    def __init__(self, root, final_folder, report_sha256):
        from music_genre.features.pretrained import PretrainedAudio
        from music_genre.models.linear import LinearProbabilityModel

        root, folder = Path(root), Path(root) / final_folder
        report = _verified_json(folder / "report.json", report_sha256)
        cache = _verified_json(root / report["feature_cache"] / "report.json", report["feature_report_sha256"])
        if sha256_file(folder / "model.npz") != report["model_sha256"]:
            raise ValueError("Final head changed")
        self.config = cache["provenance"]["config"]
        self.extractor = PretrainedAudio(self.config, root)  # checks the checkpoint hashes
        if self.extractor.families != cache["feature_families"]:
            raise ValueError("Feature families differ from the development cache")
        self.columns = [i for i, f in enumerate(self.extractor.families) if f == report["family"]]
        self.model = LinearProbabilityModel.load(folder / "model.npz")
        if len(self.columns) != report["columns"] or not np.array_equal(self.model.classes, np.arange(8)):
            raise ValueError("Unexpected head dimensions or class order")

    def predict(self, path):
        """Window probabilities and the block-12 features they come from."""
        windows = _windows(path, self.config)
        with threadpool_limits(limits=1):  # as the extraction workers that built the cache
            x = np.stack([self.extractor.extract(w) for w in windows])[:, self.columns]
        return self.model.predict_proba(x), x


class CnnV2Predictor:
    def __init__(self, root, run_folder, report_sha256):
        import torch
        from safetensors.torch import load_file

        from music_genre.models.cnn_v2 import ReinforcedAudioCNN

        folder = Path(root) / run_folder
        run = _verified_json(folder / "report.json", report_sha256)
        if run["status"] != "complete" or sha256_file(folder / "model.safetensors") != run["model_sha256"]:
            raise ValueError("Incomplete or changed CNN export")
        self.config = run["config"]
        if not self.config.get("batch_norm") or self.config["memory_format"] != "channels_last":
            raise ValueError("Not a reinforced CNN export")
        self.model = ReinforcedAudioCNN(len(run["labels"])).to(memory_format=torch.channels_last)
        self.model.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
        self.model.eval().requires_grad_(False)

    def predict(self, path):
        import torch

        from music_genre.features.logmel import log_mel
        from music_genre.training.cnn import predict_windows

        windows = _windows(path, self.config)
        with threadpool_limits(limits=1):
            x = np.stack([log_mel(w, self.config["logmel"]) for w in windows]).astype(np.float32)
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(self.config["threads"])
            with threadpool_limits(limits=self.config["threads"]):
                probabilities = predict_windows(self.model, x, np.arange(len(x)), self.config)
        finally:
            torch.set_num_threads(previous)
        return probabilities.astype(np.float64), x
