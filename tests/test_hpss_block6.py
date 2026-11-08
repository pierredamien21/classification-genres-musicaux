import json
from pathlib import Path

import numpy as np
import pytest

from music_genre.audio.processing import AudioWindow
from music_genre.features.hpss import MusicalHpssViT
from music_genre.features.hpss_block6 import MusicalHpssBlock6

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/vit_block6_hpss.json").read_text(encoding="utf-8"))


def test_hpss_block6_matches_same_channels_at_sixth_block_of_reference():
    if not (ROOT / "artifacts/pretrained/vit_tiny/model.safetensors").exists():
        pytest.skip("Local checkpoint required; no download")
    import torch

    reference_config = json.loads((ROOT / "configs/vit_hpss.json").read_text(encoding="utf-8"))
    full, sixth = MusicalHpssViT(reference_config, ROOT), MusicalHpssBlock6(CONFIG, ROOT)
    signal = np.random.default_rng(42).normal(0, .1, 160000).astype(np.float32)
    window = AudioWindow(signal, 16000, 10., len(signal))
    captured = []

    def capture(_module, _inputs, output):
        captured.append(output.detach().clone())

    hook = full.encoder.model.blocks[5].register_forward_hook(capture)
    last = full.extract(window)
    hook.remove()
    with torch.inference_mode():
        expected = full.encoder.model.norm(captured[0])[:, full.encoder.model.num_prefix_tokens:].mean(dim=1)[0].numpy()
    actual = sixth.extract(window)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)
    assert not np.allclose(actual, last)
    assert actual.shape == (192,) and np.isfinite(actual).all()
    np.testing.assert_array_equal(sixth.spectrograms(window), full.spectrograms(window))
    assert not sixth.encoder.model.training
    assert not any(p.requires_grad for p in sixth.encoder.model.parameters())


@pytest.mark.parametrize("field,value", [
    ("feature_block", 12), ("feature_block", True),
    ("terminal_norm", "none"), ("pooling", "last_block_mean_patch_tokens_exclude_prefix"),
])
def test_unsupported_protocol_fails_before_checkpoint_loading(field, value):
    config = {**CONFIG, field: value}
    with pytest.raises(ValueError, match="block-six HPSS protocol"):
        MusicalHpssBlock6(config, ROOT / "missing-checkpoint-root")
