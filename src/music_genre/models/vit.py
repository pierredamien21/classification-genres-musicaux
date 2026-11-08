"""Local frozen ViT feature extraction; never downloads weights implicitly."""

import hashlib
import json
from pathlib import Path

import numpy as np
import timm
import torch
from safetensors.torch import load_file
from torch import nn
from torch.nn import functional as F


class FrozenViT:
    """Frozen pinned checkpoint, optionally truncated after `block` (one-based).

    The terminal LayerNorm is kept after the selected block. Inputs are one
    log-Mel image (repeated on three channels) or three distinct channels.
    """

    def __init__(self, directory: Path, block: int = 12):
        if type(block) is not int or not 1 <= block <= 12:
            raise ValueError("Expected a one-based block in 1..12")
        config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
        provenance = json.loads((directory / "provenance.json").read_text(encoding="utf-8"))
        for filename in ("config.json", "model.safetensors"):
            digest = hashlib.sha256((directory / filename).read_bytes()).hexdigest()
            if digest != provenance["sha256"][filename]:
                raise ValueError(f"Checkpoint checksum mismatch: {filename}")
        self.model = timm.create_model(
            config["architecture"], pretrained=False, num_classes=config["num_classes"]
        )
        self.model.load_state_dict(load_file(str(directory / "model.safetensors")), strict=True)
        self.model.reset_classifier(0)
        if len(self.model.blocks) != 12:
            raise ValueError("Expected the pinned twelve-block ViT")
        if block < 12:
            self.model.blocks = nn.Sequential(*list(self.model.blocks.children())[:block])
        self.model.eval().requires_grad_(False)
        self.mean = torch.tensor(config["pretrained_cfg"]["mean"]).view(1, 3, 1, 1)
        self.std = torch.tensor(config["pretrained_cfg"]["std"]).view(1, 3, 1, 1)
        self.provenance = provenance

    @torch.inference_mode()
    def extract(self, spectrogram: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
        if (spectrogram.ndim not in (2, 3) or (spectrogram.ndim == 3 and spectrogram.shape[0] != 3)
                or not np.isfinite(spectrogram).all() or db_max <= db_min):
            raise ValueError("Expected a finite 2D or 3-channel spectrogram and an increasing dB interval")
        values = np.clip((spectrogram - db_min) / (db_max - db_min), 0, 1)
        tensor = torch.from_numpy(np.asarray(values, dtype=np.float32))
        tensor = tensor.view(1, 1, *values.shape) if values.ndim == 2 else tensor.unsqueeze(0)
        # Resize one channel then repeat, as in V1: identical to resizing three copies.
        tensor = F.interpolate(tensor, size=(224, 224), mode="bicubic", align_corners=False,
                               antialias=True).clamp(0, 1)
        if values.ndim == 2:
            tensor = tensor.repeat(1, 3, 1, 1)
        tensor = (tensor - self.mean) / self.std
        tokens = self.model.forward_features(tensor)
        result = tokens[:, self.model.num_prefix_tokens:].mean(dim=1)
        if not torch.isfinite(result).all():
            raise ValueError("Non-finite ViT embedding")
        return result[0].numpy().copy()
