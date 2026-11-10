"""V2 phase 3, point 2: the V1 ViT-Tiny fed the native log-Mel grid instead of a 224 x 224 resize.

Position embeddings follow AST (Gong et al. 2021, src/models/ast_models.py): the pretrained 14 x 14
grid is cut from the middle on an axis needing at most 14 patches, bilinearly interpolated otherwise.
Unlike AST, patches do not overlap (the pretrained stride 16) to keep the CPU cost: a 128 x 622
log-Mel gives 8 x 38 patches, the last 14 frames (0.22 s) are not covered.
models/vit.py is pinned by the V1 lock, so this lives here and reuses its verified loader.
"""

from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from music_genre.models.vit import FrozenViT

BLOCKS = (6, 12)


class FineTunedViT(torch.nn.Module):
    """Native-grid ViT kept up to `depth` blocks; only the last `trained` blocks, the terminal norm and a
    linear head learn (phase 3, point 4). Input: (B, 1, 128, T) log-Mel scaled to [-1, 1], which is the
    checkpoint normalization (mean = std = 0.5) of the gray image repeated on three channels.
    """

    def __init__(self, directory: Path, depth: int = 6, trained: int = 2, classes: int = 8):
        super().__init__()
        if not 1 <= trained <= depth <= 12:
            raise ValueError("Expected 1 <= trained <= depth <= 12")
        native = NativeViT(directory)
        model = native.model
        if native.mean.flatten().tolist() != [0.5] * 3 or native.std.flatten().tolist() != [0.5] * 3:
            raise ValueError("Expected the mean = std = 0.5 checkpoint normalization")
        self.patch = model.patch_embed.proj
        self.register_buffer("grid", native.grid.detach().clone())
        self.cls = torch.nn.Parameter(model.cls_token + model.pos_embed[:, :1], requires_grad=False)
        self.blocks = torch.nn.Sequential(*list(model.blocks)[:depth])
        self.norm = model.norm
        self.head = torch.nn.Linear(model.embed_dim, classes)
        for module in (*self.blocks[depth - trained:], self.norm):
            module.requires_grad_(True)

    def pooled(self, x):
        patches = self.patch(x.expand(-1, 3, -1, -1))
        tokens = patches.flatten(2).transpose(1, 2) + ast_positions(self.grid, *patches.shape[2:])
        tokens = torch.cat([self.cls.expand(len(tokens), -1, -1), tokens], dim=1)
        return self.norm(self.blocks(tokens))[:, 1:].mean(dim=1)

    def forward(self, x):
        return self.head(self.pooled(x))


def ast_positions(grid, f_dim, t_dim):
    """(1, D, hw, hw) patch position embeddings -> (1, f_dim * t_dim, D), row-major like the patches."""
    hw = grid.shape[-1]
    if t_dim <= hw:
        grid = grid[..., hw // 2 - t_dim // 2: hw // 2 - t_dim // 2 + t_dim]
    else:
        grid = F.interpolate(grid, size=(hw, t_dim), mode="bilinear")
    if f_dim <= hw:
        grid = grid[:, :, hw // 2 - f_dim // 2: hw // 2 - f_dim // 2 + f_dim, :]
    else:
        grid = F.interpolate(grid, size=(f_dim, t_dim), mode="bilinear")
    return grid.flatten(2).transpose(1, 2)


class NativeViT:
    """Mean patch tokens after blocks 6 and 12, each through the terminal LayerNorm (V1 convention)."""

    def __init__(self, directory: Path):
        frozen = FrozenViT(directory, 12)  # checksum-verified, frozen, twelve blocks
        self.model, self.mean, self.std = frozen.model, frozen.mean, frozen.std
        if self.model.num_prefix_tokens != 1 or self.model.no_embed_class:
            raise ValueError("Expected one class token with its own position")
        side = int((self.model.pos_embed.shape[1] - 1) ** 0.5)
        self.grid = self.model.pos_embed[:, 1:].reshape(1, side, side, -1).permute(0, 3, 1, 2)

    @torch.inference_mode()
    def extract(self, spectrogram: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
        if spectrogram.ndim != 2 or not np.isfinite(spectrogram).all() or db_max <= db_min:
            raise ValueError("Expected a finite 2D spectrogram and an increasing dB interval")
        values = np.clip((spectrogram - db_min) / (db_max - db_min), 0, 1).astype(np.float32)
        x = (torch.from_numpy(values).view(1, 1, *values.shape).repeat(1, 3, 1, 1) - self.mean) / self.std
        patches = self.model.patch_embed.proj(x)  # (1, D, f, t); the conv drops incomplete edge patches
        tokens = patches.flatten(2).transpose(1, 2) + ast_positions(self.grid, *patches.shape[2:])
        x = torch.cat([self.model.cls_token + self.model.pos_embed[:, :1], tokens], dim=1)
        outputs = []
        for position, block in enumerate(self.model.blocks, start=1):
            x = block(x)
            if position in BLOCKS:
                outputs.append(self.model.norm(x)[:, 1:].mean(dim=1))
        result = torch.cat(outputs, dim=1)
        if not torch.isfinite(result).all():
            raise ValueError("Non-finite ViT embedding")
        return result[0].numpy().copy()
