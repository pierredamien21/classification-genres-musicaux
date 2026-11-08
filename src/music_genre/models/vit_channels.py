"""Three distinct spectrogram channels with the unchanged frozen block-12 ViT."""

import numpy as np
import torch
from torch.nn import functional as F

from music_genre.models.vit import FrozenViT


class FrozenViTChannels(FrozenViT):
    @torch.inference_mode()
    def extract(self, spectrogram: np.ndarray, db_min: float, db_max: float) -> np.ndarray:
        if (spectrogram.ndim != 3 or spectrogram.shape[0] != 3
                or not np.isfinite(spectrogram).all() or db_max <= db_min):
            raise ValueError("Expected finite 3-channel spectrogram and an increasing dB interval")
        values = np.clip((spectrogram - db_min) / (db_max - db_min), 0, 1)
        tensor = torch.from_numpy(np.asarray(values, dtype=np.float32)).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(224, 224), mode="bicubic", align_corners=False,
                               antialias=True).clamp(0, 1)
        tensor = (tensor - self.mean) / self.std
        tokens = self.model.forward_features(tensor)
        result = tokens[:, self.model.num_prefix_tokens:].mean(dim=1)
        if not torch.isfinite(result).all():
            raise ValueError("Non-finite ViT embedding")
        return result[0].numpy().copy()
