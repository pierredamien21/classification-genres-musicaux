"""Window features from the frozen audio-pretrained encoders (configs/<model>.json, V2 phase 3)."""

from pathlib import Path

import torch

from music_genre.data.audit import sha256_file
from music_genre.features.logmel import log_mel
from music_genre.models.audio_pretrained import TRANSFORMER_LAYERS, Cnn14Encoder, TransformerEncoder
from music_genre.models.vit_native import BLOCKS, NativeViT

MODELS = ("ast", "maest", "panns_cnn14_16k", "vit_tiny_native")


class PretrainedAudio:
    def __init__(self, config: dict, root: Path):
        if config["model"] not in MODELS:
            raise ValueError(f"Unknown pretrained model: {config['model']}")
        self.config = config
        torch.set_num_threads(config["threads_per_worker"])
        directory = (root / config["checkpoint_directory"]).resolve()
        if not directory.is_relative_to((root / "artifacts/pretrained").resolve()):
            raise ValueError("Checkpoint outside pretrained directory")
        for filename, expected in config["checkpoint_sha256"].items():
            if sha256_file(directory / filename) != expected:
                raise ValueError(f"Pinned checkpoint changed: {filename}")
        if config["model"] == "vit_tiny_native":
            self.encoder = NativeViT(directory)
            self.names = [f"vit_native_block{b}_{i:03d}" for b in BLOCKS for i in range(192)]
            self.families = [f"block{b}" for b in BLOCKS for _ in range(192)]
        elif config["model"] == "panns_cnn14_16k":
            (weights,) = config["checkpoint_sha256"]
            self.encoder = Cnn14Encoder(directory / weights)
            self.names = [f"cnn14_embedding_{i:04d}" for i in range(2048)]
            self.families = ["embedding"] * 2048
        else:
            self.encoder = TransformerEncoder(directory, config["model"])
            self.names, self.families = [], []
            for layer in TRANSFORMER_LAYERS:
                for part in ("cls", "dist", "patch_mean"):
                    self.names += [f"{config['model']}_l{layer}_{part}_{i:03d}" for i in range(768)]
                    self.families += [f"layer{layer}"] * 768

    def extract(self, window):
        if window.sample_rate != self.config["sample_rate"]:
            raise ValueError("Unexpected audio sample rate")
        if len(window.samples) != round(window.sample_rate * self.config["window_seconds"]):
            raise ValueError("Expected fixed-duration padded window")
        if self.config["model"] == "vit_tiny_native":
            return self.encoder.extract(log_mel(window, self.config["logmel"]), self.config["db_min"],
                                        self.config["db_max"])
        return self.encoder.forward(window.samples)[0]
