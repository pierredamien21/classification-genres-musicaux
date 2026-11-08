"""Frozen image-pretrained ViT applied to explicitly configured audio windows.

One extractor covers every V1 variant, read from the cached feature configuration:
- channels "repeat_grayscale_three_times" (log-Mel) or "original_harmonic_percussive" (HPSS);
- no `feature_block` (all twelve blocks) or `feature_block` 1..12 with the terminal LayerNorm.
"""

from pathlib import Path

import torch

from music_genre.data.audit import sha256_file
from music_genre.features.hpss import hpss_log_mel
from music_genre.features.logmel import log_mel
from music_genre.models.vit import FrozenViT

CHANNELS = ("repeat_grayscale_three_times", "original_harmonic_percussive")
LAST_BLOCK = "last_block_mean_patch_tokens_exclude_prefix"
SELECTED_BLOCK = "selected_block_mean_patch_tokens_exclude_prefix"
TERMINAL_NORM = "same_frozen_checkpoint_layernorm_after_selected_block"


def vit_block(config: dict) -> int:
    """Validate the representation protocol and return the one-based block to keep."""
    if config.get("channels") not in CHANNELS:
        raise ValueError("Unsupported ViT channel protocol")
    if config["channels"] == CHANNELS[1] and (
            config["hpss"]["domain"] != "complex_stft_before_mel" or config["logmel"]["power"] != 2.0):
        raise ValueError("Unsupported HPSS representation protocol")
    if "feature_block" not in config:
        if config.get("pooling") != LAST_BLOCK:
            raise ValueError("Unsupported ViT pooling protocol")
        return 12
    block = config["feature_block"]
    if (type(block) is not int or not 1 <= block <= 12 or config.get("pooling") != SELECTED_BLOCK
            or config.get("terminal_norm") != TERMINAL_NORM):
        raise ValueError("Unsupported selected-block ViT protocol")
    return block


class MusicalViT:
    def __init__(self, config: dict, root: Path):
        block = vit_block(config)
        self.config = config
        torch.set_num_threads(config["threads_per_worker"])
        directory = (root / config["checkpoint_directory"]).resolve()
        if not directory.is_relative_to((root / "artifacts/pretrained").resolve()):
            raise ValueError("Checkpoint outside pretrained directory")
        for filename, expected in config["checkpoint"]["sha256"].items():
            if sha256_file(directory / filename) != expected:
                raise ValueError(f"Pinned checkpoint changed: {filename}")
        self.encoder = FrozenViT(directory, block)
        self.names = [f"vit_patch_mean_{i:03d}" for i in range(192)]
        self.families = ["vit"] * len(self.names)

    def spectrograms(self, window):
        if window.sample_rate != self.config["sample_rate"]:
            raise ValueError("Unexpected audio sample rate")
        if len(window.samples) != round(window.sample_rate * self.config["window_seconds"]):
            raise ValueError("Expected fixed-duration padded window")
        if self.config["channels"] == CHANNELS[1]:
            return hpss_log_mel(window, self.config["logmel"], self.config["hpss"])
        return log_mel(window, self.config["logmel"])

    def extract(self, window):
        return self.encoder.extract(
            self.spectrograms(window), self.config["db_min"], self.config["db_max"]
        )
