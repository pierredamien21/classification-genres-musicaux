"""Remaining pinned linear comparators; historical inference sources stay intact."""

import hashlib
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from music_genre.data.audit import sha256_file
from music_genre.features.descriptors import MusicalDescriptors
from music_genre.features.hpss import MusicalHpssViT
from music_genre.features.vit import MusicalViT
from music_genre.inference.predict import CandidatePredictor, _inside, _verified_json
from music_genre.models.linear import LinearProbabilityModel


class AlignedFusion:
    """Extract both representations from the very same AudioWindow instance.

    The cache concatenator requires train/validation labels; inference must not
    invent those. Here alignment is by construction, with pinned column metadata.
    """

    def __init__(self, config, root):
        if (config["operation"] != "concatenate_columns" or config["order"] != ["vit", "descriptors"]
                or config["excluded_family"] is not None
                or config["standardization"] != "none_in_cache_train_only_in_classifier"
                or any(config["vit"][k] != config["descriptors"][k]
                       for k in ("sample_rate", "window_seconds", "offsets_seconds"))):
            raise ValueError("Fusion representation order or windows differ")
        self.descriptors = MusicalDescriptors(config["descriptors"])
        if config["descriptor_columns"] != list(range(len(self.descriptors.names))):
            raise ValueError("Fusion descriptor columns differ")
        self.vit = MusicalViT(config["vit"], root)
        self.names = self.vit.names + [f"descriptor/{name}" for name in self.descriptors.names]
        self.families = self.vit.families + self.descriptors.families

    def extract(self, window):
        return np.concatenate((self.vit.extract(window), self.descriptors.extract(window)))


class LinearComparatorPredictor(CandidatePredictor):
    """Load a prepared comparator and reuse the same unlabelled audio contract."""

    def __init__(self, root, system, preparation="reports/PREPARATION_EVALUATION_FINALE.json"):
        if system not in ("descriptor", "vit", "fusion", "vit_hpss"):
            raise ValueError("Unsupported linear comparator")
        root = Path(root).resolve()
        prepared = json.loads(_inside(root, preparation, "reports").read_text("utf-8"))
        pin = prepared["runs"][system]
        folder = _inside(root, pin["run"]["folder"], "runs")
        run = _verified_json(folder / "report.json", pin["run"]["report_sha256"])
        cache = _inside(root, pin["feature_cache"], "data/features")
        features = _verified_json(cache / "report.json", pin["feature_report_sha256"])
        if (run["feature_cache"] != pin["feature_cache"]
                or run["feature_report_sha256"] != pin["feature_report_sha256"]):
            raise ValueError("Inconsistent feature provenance")
        config = features["provenance"]["config"]
        self.labels = list(run["labels"])
        model_path = _inside(root, pin["model_path"], "runs")
        if (model_path != folder / "model.npz" or pin["model_sha256"] != run["model_sha256"]
                or sha256_file(model_path) != pin["model_sha256"]
                or self.labels != prepared["labels"] or len(self.labels) != 8
                or len(set(self.labels)) != 8
                or features["provenance"]["labels"] != self.labels):
            raise ValueError("Inconsistent comparator export")
        self.model = LinearProbabilityModel.load(model_path)
        if (not np.array_equal(self.model.classes, np.arange(8))
                or len(self.model.mean) != len(features["feature_names"])):
            raise ValueError("Unexpected feature dimensions or class order")
        if system == "descriptor":
            self.extractor = MusicalDescriptors(config)
        elif system == "fusion":
            self.extractor = AlignedFusion(config, root)
        elif system == "vit":
            self.extractor = MusicalViT(config, root)
        else:
            self.extractor = MusicalHpssViT(config, root)
        self.config = config["vit"] if system == "fusion" else config
        if (self.extractor.names != features["feature_names"]
                or self.extractor.families != features["feature_families"]):
            raise ValueError("Feature order or families differ from trained export")
        identity = {"model_sha256": pin["model_sha256"], "labels": self.labels,
                    "representation": config}
        self.model_id = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()

    def predict_file(self, path, *, mode="fma"):
        # Match the historical extraction workers' BLAS/OpenMP configuration.
        # The bounded context restores the caller's settings after prediction.
        with threadpool_limits(limits=1):
            return super().predict_file(path, mode=mode)
