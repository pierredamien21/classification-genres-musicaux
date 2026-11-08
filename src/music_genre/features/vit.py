"""Frozen image-pretrained ViT applied to explicitly configured audio windows."""

from pathlib import Path

import torch

from music_genre.data.audit import sha256_file
from music_genre.features.logmel import log_mel
from music_genre.models.vit import FrozenViT


class MusicalViT:
    def __init__(self, config: dict, root: Path):
        self.config = config
        torch.set_num_threads(config["threads_per_worker"])
        directory = (root / config["checkpoint_directory"]).resolve()
        if not directory.is_relative_to((root / "artifacts/pretrained").resolve()):
            raise ValueError("Checkpoint outside pretrained directory")
        for filename, expected in config["checkpoint"]["sha256"].items():
            if sha256_file(directory / filename) != expected:
                raise ValueError(f"Pinned checkpoint changed: {filename}")
        self.encoder = FrozenViT(directory)
        self.names = [f"vit_patch_mean_{i:03d}" for i in range(192)]
        self.families = ["vit"] * len(self.names)

    def extract(self, window):
        if window.sample_rate != self.config["sample_rate"]:
            raise ValueError("Unexpected audio sample rate")
        if len(window.samples) != round(window.sample_rate * self.config["window_seconds"]):
            raise ValueError("Expected fixed-duration padded window")
        return self.encoder.extract(
            log_mel(window, self.config["logmel"]), self.config["db_min"], self.config["db_max"]
        )
