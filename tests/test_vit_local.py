from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_local_frozen_features_are_finite_and_repeatable():
    weights = ROOT / "artifacts/pretrained/vit_tiny"
    if not (weights / "model.safetensors").exists():
        pytest.skip("Local checkpoint required; this test never downloads weights")
    import torch

    from music_genre.models.vit import FrozenViT

    torch.set_num_threads(4)
    extractor = FrozenViT(weights)
    mel = np.linspace(-80, 20, 128 * 622, dtype=np.float32).reshape(128, 622)
    first = extractor.extract(mel, -80, 20)
    second = extractor.extract(mel, -80, 20)
    assert first.shape == (192,)
    assert np.isfinite(first).all()
    np.testing.assert_allclose(first, second, atol=1e-6)
    assert not any(parameter.requires_grad for parameter in extractor.model.parameters())
