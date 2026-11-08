"""HPSS reference inputs with the frozen six-block prefix and final LayerNorm."""

from torch import nn

from music_genre.features.hpss import MusicalHpssViT


class MusicalHpssBlock6(MusicalHpssViT):
    def __init__(self, config, root):
        if (type(config.get("feature_block")) is not int or config["feature_block"] != 6
                or config.get("pooling") != "selected_block_mean_patch_tokens_exclude_prefix"
                or config.get("terminal_norm") != "same_frozen_checkpoint_layernorm_after_selected_block"):
            raise ValueError("Unsupported block-six HPSS protocol")
        # Reuse the pinned HPSS initialization, then retain only the prefix.
        # The original reference code and its cache provenance stay unchanged.
        super().__init__({**config, "pooling": "last_block_mean_patch_tokens_exclude_prefix"}, root)
        self.config = config
        if len(self.encoder.model.blocks) != 12:
            raise ValueError("Expected the pinned twelve-block ViT")
        self.encoder.model.blocks = nn.Sequential(*list(self.encoder.model.blocks.children())[:6])
        self.encoder.model.eval().requires_grad_(False)
