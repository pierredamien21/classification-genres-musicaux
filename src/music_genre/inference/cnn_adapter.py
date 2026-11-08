"""Pinned native-resolution CNN inference; no fitting or seed selection."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from threadpoolctl import threadpool_limits

from music_genre.data.audit import sha256_file
from music_genre.features.logmel import log_mel
from music_genre.inference.predict import CandidatePredictor, _inside, _verified_json
from music_genre.models.cnn import SmallAudioCNN
from music_genre.training.cnn import predict_windows


class NativeLogMel:
    def __init__(self, config):
        self.config = config

    def extract(self, window):
        if (window.sample_rate != self.config["sample_rate"]
                or len(window.samples) != round(window.sample_rate * self.config["window_seconds"])):
            raise ValueError("Unexpected CNN window geometry")
        with threadpool_limits(limits=1):
            return log_mel(window, self.config["logmel"])


class CnnProbabilities:
    def __init__(self, model, config):
        self.native = model
        self.config = config

    def predict_proba(self, matrix):
        with threadpool_limits(limits=self.config["threads"]):
            probabilities = predict_windows(self.native, matrix, np.arange(len(matrix)), self.config)
        # Historical aggregate_tracks casts windows to float64 before averaging.
        return probabilities.astype(np.float64)


class CnnPredictor(CandidatePredictor):
    """Predict a file with one of the three prepared CNN exports, never an ensemble."""

    def __init__(self, root, seed, preparation="reports/PREPARATION_EVALUATION_FINALE.json"):
        if type(seed) is not int or seed not in (42, 43, 44):
            raise ValueError("Expected prepared CNN seed 42, 43 or 44")
        root = Path(root).resolve()
        prepared = json.loads(_inside(root, preparation, "reports").read_text("utf-8"))
        pin = prepared["runs"][f"cnn20_seed{seed}"]
        folder = _inside(root, pin["run"]["folder"], "runs")
        run = _verified_json(folder / "report.json", pin["run"]["report_sha256"])
        cache = _inside(root, pin["feature_cache"], "data/features")
        features = _verified_json(cache / "report.json", pin["feature_report_sha256"])
        model_path = _inside(root, pin["model_path"], "runs")
        self.config, self.labels = run["config"], list(run["labels"])
        self.selected_epoch = run["selected_epoch"]
        if (model_path != folder / "model.safetensors" or pin["model_sha256"] != run["model_sha256"]
                or sha256_file(model_path) != pin["model_sha256"]
                or self.labels != prepared["labels"] or len(self.labels) != 8 or len(set(self.labels)) != 8
                or run["feature_cache"] != pin["feature_cache"]
                or run["feature_report_sha256"] != pin["feature_report_sha256"]
                or self.config["seed"] != seed or self.selected_epoch != pin["selected_epoch"]
                or any(self.config[k] != value for k, value in features["provenance"]["config"].items())):
            raise ValueError("Inconsistent CNN export")
        architecture = {"input_shape": [1, 128, 622], "channels": [16, 32, 64],
                        "kernel_size": 3, "strides": [2, 1, 1], "padding": 1,
                        "pooling": "max2_each_block_then_global_average", "memory_format": "channels_last"}
        if any(self.config[k] != value for k, value in architecture.items()):
            raise ValueError("Unsupported CNN architecture")
        native = SmallAudioCNN(len(self.labels)).to(memory_format=torch.channels_last)
        native.load_state_dict(load_file(str(model_path)), strict=True)
        native.eval().requires_grad_(False)
        self.model = CnnProbabilities(native, self.config)
        self.extractor = NativeLogMel(self.config)
        identity = {"model_sha256": pin["model_sha256"], "labels": self.labels,
                    "config": self.config, "selected_epoch": self.selected_epoch}
        self.model_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def predict_file(self, path, *, mode="fma"):
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self.config["threads"])
            return super().predict_file(path, mode=mode)
        finally:
            torch.set_num_threads(previous_threads)
