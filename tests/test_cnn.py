import numpy as np
import pytest
import torch
from safetensors.torch import load_file, save_file

from music_genre.models.cnn import SmallAudioCNN, normalize_logmel
from music_genre.training.cnn import epoch_order, load_checkpoint, make_batch, save_checkpoint


def test_native_cnn_forward_backward_and_export(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(42)
    model = SmallAudioCNN().to(memory_format=torch.channels_last)
    x = torch.rand(2, 1, 128, 622).contiguous(memory_format=torch.channels_last)
    logits = model(x)
    assert logits.shape == (2, 8)
    assert sum(p.numel() for p in model.parameters()) == 23816
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 7])).backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    model.eval()
    path = tmp_path / "model.safetensors"
    save_file({k: v.contiguous() for k, v in model.state_dict().items()}, str(path))
    restored = SmallAudioCNN().to(memory_format=torch.channels_last)
    restored.load_state_dict(load_file(str(path)))
    restored.eval()
    torch.testing.assert_close(model(x), restored(x), rtol=0, atol=0)


def test_cnn_normalization_is_fixed_and_batch_has_no_side_effects():
    values = torch.tensor([-100., -80., -30., 20., 40.])
    torch.testing.assert_close(normalize_logmel(values), torch.tensor([-1., -1., 0., 1., 1.]))
    raw = np.full((2, 128, 622), -30., dtype=np.float32)
    raw[1] = 20
    before = raw.copy()
    batch = make_batch(raw, [1, 0], {"db_min": -80, "db_max": 20})
    assert batch.shape == (2, 1, 128, 622)
    assert torch.all(batch[0] == 1) and torch.all(batch[1] == 0)
    np.testing.assert_array_equal(raw, before)
    raw[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        make_batch(raw, [0], {"db_min": -80, "db_max": 20})


def test_epoch_order_is_reproducible_complete_and_changes_each_epoch():
    positions = np.arange(30)
    a, b = epoch_order(positions, 42, 1), epoch_order(positions, 42, 2)
    np.testing.assert_array_equal(a, epoch_order(positions, 42, 1))
    np.testing.assert_array_equal(np.sort(a), positions)
    assert not np.array_equal(a, b)


def test_checkpoint_resumes_optimizer_exactly_and_rejects_changed_identity(tmp_path):
    torch.manual_seed(5)
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    x, y = torch.rand(4, 3), torch.tensor([0, 1, 0, 1])

    def step(m, opt):
        opt.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(m(x), y).backward()
        opt.step()

    step(model, optimizer)
    identity = {"seed": 5, "cache_sha256": "verified"}
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, {"identity": identity, "model": model.state_dict(),
                           "optimizer": optimizer.state_dict(), "epoch": 1})
    step(model, optimizer)
    loaded = load_checkpoint(path, identity)
    resumed = torch.nn.Linear(3, 2)
    resumed.load_state_dict(loaded["model"])
    resumed_optimizer = torch.optim.AdamW(resumed.parameters(), lr=0.001)
    resumed_optimizer.load_state_dict(loaded["optimizer"])
    step(resumed, resumed_optimizer)
    for expected, actual in zip(model.parameters(), resumed.parameters(), strict=True):
        torch.testing.assert_close(expected, actual, atol=0, rtol=0)
    with pytest.raises(ValueError, match="changed"):
        load_checkpoint(path, {**identity, "cache_sha256": "changed"})
