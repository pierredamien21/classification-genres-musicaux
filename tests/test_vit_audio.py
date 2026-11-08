import json
from pathlib import Path

import numpy as np
import pytest

from music_genre.audio.processing import AudioWindow

ROOT = Path(__file__).resolve().parents[1]


def test_audio_vit_contract_and_silence():
    if not (ROOT / "artifacts/pretrained/vit_tiny/model.safetensors").exists():
        pytest.skip("Local checkpoint required; no downloads")
    from music_genre.features.vit import MusicalViT

    config = json.loads((ROOT / "configs/vit_frozen.json").read_text())
    extractor = MusicalViT(config, ROOT)
    window = AudioWindow(np.zeros(160000, dtype=np.float32), 16000, 0.0, 160000)
    result = extractor.extract(window)
    assert result.shape == (len(extractor.names),) == (192,)
    assert np.isfinite(result).all()
    assert not extractor.encoder.model.training
    assert not any(p.requires_grad for p in extractor.encoder.model.parameters())
    with pytest.raises(ValueError, match="sample rate"):
        extractor.extract(AudioWindow(window.samples, 22050, 0.0, 160000))
    with pytest.raises(ValueError, match="fixed-duration"):
        extractor.extract(AudioWindow(window.samples[:1000], 16000, 0.0, 1000))


def test_pinned_checkpoint_rejects_changed_file(tmp_path):
    from music_genre.features.vit import MusicalViT

    config = json.loads((ROOT / "configs/vit_frozen.json").read_text())
    target = tmp_path / config["checkpoint_directory"]
    target.mkdir(parents=True)
    (target / "config.json").write_text("{}")
    with pytest.raises(ValueError, match="Pinned checkpoint changed"):
        MusicalViT(config, tmp_path)
