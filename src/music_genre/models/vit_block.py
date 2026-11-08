"""Frozen prefix of the same ViT, followed by its unchanged terminal LayerNorm."""

from pathlib import Path

from torch import nn

from music_genre.models.vit import FrozenViT


class FrozenViTBlock(FrozenViT):
    def __init__(self, directory: Path, block: int):
        if type(block) is not int or not 1 <= block <= 12:
            raise ValueError("Expected a one-based block in 1..12")
        super().__init__(directory)
        if len(self.model.blocks) != 12:
            raise ValueError("Expected the pinned twelve-block ViT")
        self.model.blocks = nn.Sequential(*list(self.model.blocks.children())[:block])
        self.model.eval().requires_grad_(False)
