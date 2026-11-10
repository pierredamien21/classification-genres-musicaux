import importlib.util
from argparse import Namespace
from pathlib import Path

import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from music_genre.models.cnn_v2 import ReinforcedAudioCNN, spec_augment

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("train_cnn", ROOT / "scripts/train_cnn.py")
train_cnn = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_cnn)

AUGMENT = {"freq_masks": 2, "freq_width": 16, "time_masks": 2, "time_width": 64}


def test_reinforced_cnn_normalizes_batches_and_exports_running_statistics(tmp_path):
    torch.manual_seed(42)
    model = ReinforcedAudioCNN().to(memory_format=torch.channels_last)
    assert sum(isinstance(m, torch.nn.BatchNorm2d) for m in model.modules()) == 3
    x = torch.rand(4, 1, 128, 622).contiguous(memory_format=torch.channels_last)
    logits = model(x)  # training mode updates the running statistics
    assert logits.shape == (4, 8)
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 1, 2, 7])).backward()
    model.eval()
    path = tmp_path / "model.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))
    restored = ReinforcedAudioCNN().to(memory_format=torch.channels_last)
    restored.load_state_dict(load_file(str(path)), strict=True)
    restored.eval()
    torch.testing.assert_close(model(x[:1]), restored(x[:1]), rtol=0, atol=0)


def test_spec_augment_zeroes_bounded_bands_reproducibly_without_touching_the_input():
    batch = torch.ones(3, 1, 128, 622)
    first = spec_augment(batch, AUGMENT, torch.Generator().manual_seed(7))
    again = spec_augment(batch, AUGMENT, torch.Generator().manual_seed(7))
    assert torch.equal(first, again) and torch.all(batch == 1)
    assert set(first.unique().tolist()) <= {0.0, 1.0}
    for example in first[:, 0]:
        masked_rows = int((example == 0).all(dim=1).sum())
        masked_columns = int((example == 0).all(dim=0).sum())
        assert masked_rows <= 2 * 16 and masked_columns <= 2 * 64
        assert torch.equal(example == 0, ((example == 0).all(dim=1)[:, None]
                                          | (example == 0).all(dim=0)[None, :]))  # whole bands only


def test_v2_runs_never_reuse_v1_pointer_names():
    def name(config, seed=None):
        return train_cnn.run_name_for(Namespace(run_name=None, config=config, seed=seed, smoke=False))
    assert name("configs/cnn.json", 43) == "cnn_seed43"
    assert name("configs/cnn_user.json") == "cnn_user"
    assert name("configs/cnn_v2.json", 43) == "cnn_v2_seed43"
    assert name("configs/vit_finetune.json", 42) == "vit_finetune_seed42"


def test_fold_split_holds_out_one_fold_and_rejects_shared_artists():
    index = [{"track_id": t, "artist_id": a} for t, a in (("1", "x"), ("1", "x"), ("2", "y"), ("3", "z"))]
    folds = {"1": {"fold": "0", "artist_id": "x"}, "2": {"fold": "1", "artist_id": "y"},
             "3": {"fold": "2", "artist_id": "z"}}
    positions = train_cnn.fold_positions(index, folds, 0)
    assert positions["validation"].tolist() == [0, 1] and positions["train"].tolist() == [2, 3]
    assert np.array_equal(train_cnn.fold_positions(index, folds, 1)["validation"], [2])
    with pytest.raises(ValueError, match="artist"):  # the cache and the fold file disagree
        train_cnn.fold_positions(index, {**folds, "3": {"fold": "2", "artist_id": "y"}}, 0)
    shared = [*index[:3], {"track_id": "3", "artist_id": "y"}]
    with pytest.raises(ValueError, match="artist"):  # one artist on both sides
        train_cnn.fold_positions(shared, {**folds, "3": {"fold": "1", "artist_id": "y"},
                                          "2": {"fold": "0", "artist_id": "y"}}, 0)
