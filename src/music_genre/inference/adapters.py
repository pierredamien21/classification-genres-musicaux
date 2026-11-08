"""Pinned comparator adapters sharing the candidate's audio prediction contract."""

import hashlib
import json
from pathlib import Path

import numpy as np
from torch import nn

from music_genre.data.audit import sha256_file
from music_genre.features.vit import MusicalViT
from music_genre.inference.predict import CandidatePredictor, _inside, _verified_json
from music_genre.models.linear import LinearProbabilityModel


class OriginalBlock6Features(MusicalViT):
    """Same log-Mel path and frozen prefix as the original cached experiment."""

    def __init__(self, config, root):
        if (type(config.get("feature_block")) is not int or config["feature_block"] != 6
                or config.get("channels") != "repeat_grayscale_three_times"
                or config.get("pooling") != "selected_block_mean_patch_tokens_exclude_prefix"
                or config.get("terminal_norm") != "same_frozen_checkpoint_layernorm_after_selected_block"):
            raise ValueError("Unsupported original block-six protocol")
        super().__init__(config, root)
        if len(self.encoder.model.blocks) != 12:
            raise ValueError("Expected the pinned twelve-block ViT")
        self.encoder.model.blocks = nn.Sequential(*list(self.encoder.model.blocks.children())[:6])
        self.encoder.model.eval().requires_grad_(False)


class OriginalBlock6Predictor(CandidatePredictor):
    """Load original block-six artifacts; inherit unchanged file/window inference.

    Only this initializer differs. Historical candidate source and its provenance
    remain unchanged. No feature matrix, labels of input tracks, or fitting needed.
    """

    def __init__(self, root, preparation="reports/PREPARATION_EVALUATION_FINALE.json"):
        root = Path(root).resolve()
        prepared = json.loads(_inside(root, preparation, "reports").read_text("utf-8"))
        pin = prepared["runs"]["vit_block6"]
        folder = _inside(root, pin["run"]["folder"], "runs")
        report = _verified_json(folder / "report.json", pin["run"]["report_sha256"])
        cache = _inside(root, report["feature_cache"], "data/features")
        features = _verified_json(cache / "report.json", report["feature_report_sha256"])
        if (pin["feature_cache"] != report["feature_cache"]
                or pin["feature_report_sha256"] != report["feature_report_sha256"]):
            raise ValueError("Inconsistent comparator feature provenance")
        self.config = features["provenance"]["config"]
        self.labels = list(report["labels"])
        model_path = _inside(root, pin["model_path"], "runs")
        if (model_path != folder / "model.npz"
                or pin["model_sha256"] != report["model_sha256"]
                or sha256_file(model_path) != report["model_sha256"]
                or self.labels != prepared["labels"] or len(self.labels) != 8
                or len(set(self.labels)) != 8):
            raise ValueError("Inconsistent comparator export")
        self.model = LinearProbabilityModel.load(model_path)
        if not np.array_equal(self.model.classes, np.arange(8)) or len(self.model.mean) != 192:
            raise ValueError("Unexpected comparator dimensions or class order")
        identity = {"model_sha256": report["model_sha256"], "labels": self.labels,
                    "representation": self.config}
        self.model_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self.extractor = OriginalBlock6Features(self.config, root)
