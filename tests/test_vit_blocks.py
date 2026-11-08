from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_intermediate_block_matches_full_model_hook_and_last_block():
    weights = ROOT / "artifacts/pretrained/vit_tiny"
    if not (weights / "model.safetensors").exists():
        pytest.skip("Local checkpoint required; no download")
    import torch

    from music_genre.models.vit import FrozenViT

    torch.set_num_threads(2)
    full, sixth, twelfth = FrozenViT(weights), FrozenViT(weights, 6), FrozenViT(weights, 12)
    captured = []

    def capture(_module, _inputs, output):
        captured.append(output.detach().clone())

    hook = full.model.blocks[5].register_forward_hook(capture)
    mel = np.linspace(-100, 30, 128 * 622, dtype=np.float32).reshape(128, 622)
    last = full.extract(mel, -80, 20)
    hook.remove()
    with torch.inference_mode():
        expected = full.model.norm(captured[0])[:, full.model.num_prefix_tokens:].mean(dim=1)[0].numpy()
    middle = sixth.extract(mel, -80, 20)
    np.testing.assert_allclose(middle, expected, rtol=0, atol=1e-6)
    np.testing.assert_allclose(twelfth.extract(mel, -80, 20), last, rtol=0, atol=1e-6)
    assert middle.shape == (192,) and np.isfinite(middle).all()
    assert not np.allclose(middle, last)
    assert len(sixth.model.blocks) == 6
    assert not sixth.model.training and not any(p.requires_grad for p in sixth.model.parameters())


@pytest.mark.parametrize("block", [0, 13, 6.5, True])
def test_invalid_block_fails_before_loading_checkpoint(block):
    from music_genre.models.vit import FrozenViT

    with pytest.raises(ValueError, match="1..12"):
        FrozenViT(Path("missing-checkpoint"), block)
