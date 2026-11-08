"""Explicit, traceable continuation of a completed epoch-limited CNN run."""

from copy import deepcopy

import torch


def extend_state(state, report, identity):
    """Preserve optimizer, RNG, patience and best epoch; only raise epoch ceiling."""
    old = report["config"]
    new = identity["config"]
    if (new["max_epochs"] <= old["max_epochs"]
            or {k: v for k, v in new.items() if k != "max_epochs"}
            != {k: v for k, v in old.items() if k != "max_epochs"}):
        raise ValueError("Only an increased max_epochs is permitted")
    if (report["status"] != "complete" or report["stop_reason"] != "max_epochs"
            or report["test_loaded"] or report["test_evaluated"]
            or state["identity"] != report["identity"]
            or report["identity"]["config"] != old
            or state["epoch"] != old["max_epochs"]
            or report["epochs_completed"] != state["epoch"]
            or state["history"] != report["history"]):
        raise ValueError("Expected an unchanged complete epoch-limited parent")
    history = state["history"]
    if [r["epoch"] for r in history] != list(range(1, state["epoch"] + 1)):
        raise ValueError("Incomplete parent history")
    best = max(history, key=lambda r: r["validation"]["macro_f1"])
    if (best["epoch"] != state["best_epoch"] or best["epoch"] != report["selected_epoch"]
            or best["validation"] != report["validation"]
            or best["validation"]["macro_f1"] != state["best_score"]
            or state["bad_epochs"] != state["epoch"] - state["best_epoch"]
            or state["bad_epochs"] >= old["patience"]):
        raise ValueError("Invalid selection or exhausted patience in parent")
    for key in ("cache", "torch_version", "numpy_version", "benchmark_sha256"):
        if identity[key] != state["identity"][key]:
            raise ValueError("Data or environment changed")
    for name, expected in state["identity"]["source_sha256"].items():
        if name != "scripts/train_cnn.py" and identity["source_sha256"].get(name) != expected:
            raise ValueError("Training implementation changed")
    result = deepcopy(state)
    result["identity"] = deepcopy(identity)
    return result


def verify_best_weights(state, exported):
    if (state["best_model"].keys() != exported.keys()
            or any(not torch.equal(state["best_model"][k], v) for k, v in exported.items())):
        raise ValueError("Checkpoint best weights differ from exported parent")
