from copy import deepcopy
from pathlib import Path

import pytest
import torch

from music_genre.training.cnn import load_checkpoint, save_checkpoint
from music_genre.training.extension import extend_state, verify_best_weights


def fixture_state():
    config = {"seed": 42, "max_epochs": 2, "patience": 3, "learning_rate": 0.001}
    identity = {"config": config, "cache": "frozen", "torch_version": "same",
                "numpy_version": "same", "benchmark_sha256": "same",
                "source_sha256": {"scripts/train_cnn.py": "old", "core.py": "same"}}
    history = [{"epoch": 1, "validation": {"macro_f1": 0.4}},
               {"epoch": 2, "validation": {"macro_f1": 0.5}}]
    state = {"identity": identity, "epoch": 2, "history": history, "best_epoch": 2,
             "best_score": 0.5, "bad_epochs": 0, "elapsed_seconds": 12.0,
             "best_model": {"weight": torch.ones(1)}}
    report = {"config": config, "identity": identity, "status": "complete",
              "stop_reason": "max_epochs", "test_loaded": False, "test_evaluated": False,
              "epochs_completed": 2, "selected_epoch": 2, "history": history,
              "validation": history[-1]["validation"]}
    new = {**deepcopy(identity), "config": {**config, "max_epochs": 4},
           "source_sha256": {"scripts/train_cnn_extended.py": "new", "core.py": "same"}}
    return state, report, new


@pytest.mark.parametrize("change", ["learning_rate", "cache", "source", "test", "history", "patience"])
def test_extension_rejects_uncontrolled_changes(change):
    state, report, new = fixture_state()
    if change == "learning_rate":
        new["config"][change] = 0.1
    elif change == "cache":
        new[change] = "other"
    elif change == "source":
        new["source_sha256"]["core.py"] = "other"
    elif change == "test":
        report["test_evaluated"] = True
    elif change == "history":
        report["history"] = []
    else:
        state["bad_epochs"] = 3
    with pytest.raises(ValueError):
        extend_state(state, report, new)


def test_extension_preserves_checkpoint_rng_optimizer_and_parent(tmp_path):
    torch.manual_seed(123)

    def network():
        return torch.nn.Sequential(torch.nn.Linear(3, 4), torch.nn.Dropout(0.3), torch.nn.Linear(4, 2))

    def step(model, optimizer):
        optimizer.zero_grad(set_to_none=True)
        loss = model(torch.rand(3, 3)).square().mean()
        loss.backward()
        optimizer.step()

    model = network()
    opt = torch.optim.AdamW(model.parameters(), lr=0.001)
    step(model, opt)
    step(model, opt)
    state, report, new = fixture_state()
    state.update(model=model.state_dict(), optimizer=opt.state_dict(), rng=torch.get_rng_state())
    checkpoint = tmp_path / "parent.pt"
    save_checkpoint(checkpoint, state)
    before = checkpoint.read_bytes()
    step(model, opt)
    expected_rng = torch.get_rng_state()
    resumed = extend_state(load_checkpoint(checkpoint, report["identity"]), report, new)
    restored = network()
    restored.load_state_dict(resumed["model"])
    restored_opt = torch.optim.AdamW(restored.parameters(), lr=0.001)
    restored_opt.load_state_dict(resumed["optimizer"])
    torch.set_rng_state(resumed["rng"])
    step(restored, restored_opt)
    for a, b in zip(model.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    assert torch.equal(expected_rng, torch.get_rng_state())
    assert checkpoint.read_bytes() == before
    assert resumed["history"] == report["history"]
    assert resumed["bad_epochs"] == state["bad_epochs"]
    resumed["history"][0]["epoch"] = 99
    assert report["history"][0]["epoch"] == 1


def test_extension_rejects_different_exported_best_weights():
    state, _, _ = fixture_state()
    verify_best_weights(state, {"weight": torch.ones(1)})
    with pytest.raises(ValueError):
        verify_best_weights(state, {"weight": torch.zeros(1)})


def test_extension_training_selection_and_export_match_original():
    root = Path(__file__).resolve().parents[1]
    original = (root / "scripts/train_cnn.py").read_text(encoding="utf-8")
    extension = (root / "scripts/train_cnn_extended.py").read_text(encoding="utf-8")
    def loop(source):
        return source[source.index("    with threadpool_limits"):source.index("    report = {")]
    assert loop(original) == loop(extension)
