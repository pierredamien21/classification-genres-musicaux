"""Freeze parent checkpoints and a twenty-epoch protocol before further training."""

import json
from datetime import UTC, datetime
from pathlib import Path

from safetensors.torch import load_file
from summarize_cnn_repetitions import read_repetitions

from music_genre.data.audit import sha256_file, write_json
from music_genre.training.cnn import load_checkpoint
from music_genre.training.extension import extend_state, verify_best_weights

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("scripts/train_cnn_extended.py", "src/music_genre/training/extension.py",
           "src/music_genre/training/cnn.py", "src/music_genre/models/cnn.py",
           "src/music_genre/evaluation/metrics.py")


def main():
    destination = ROOT / "configs/cnn_extension.json"
    if destination.exists():
        raise FileExistsError("Keep the frozen extension protocol; do not overwrite it")
    old_plan, pointers, reports, _ = read_repetitions()
    sources = {name: sha256_file(ROOT / name) for name in SOURCES}
    parents, costs = [], []
    for pointer, report in zip(pointers, reports, strict=True):
        folder = ROOT / pointer["folder"]
        state = load_checkpoint(folder / "checkpoint.pt", report["identity"])
        config = {**report["config"], "max_epochs": 20}
        identity = {**report["identity"], "config": config, "source_sha256": sources}
        extend_state(state, report, identity)
        verify_best_weights(state, load_file(str(folder / "model.safetensors")))
        # Measured full epochs include cache reads, validation and aggregation.
        epoch_seconds = max(row["seconds"] for row in report["history"])
        extra = 1.5 * epoch_seconds * (20 - state["epoch"]) + 120
        if state["elapsed_seconds"] + extra > config["budget_seconds"]:
            raise ValueError("Estimated continuation exceeds the existing time budget")
        parents.append({**pointer, "seed": config["seed"],
                        "checkpoint_sha256": sha256_file(folder / "checkpoint.pt")})
        costs.append({"seed": config["seed"], "max_observed_epoch_seconds": epoch_seconds,
                      "estimated_additional_seconds_with_margin": extra})
    plan = {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(),
        "seeds": old_plan["seeds"], "max_epochs": 20, "parents": parents,
        "parent_plan_sha256": sha256_file(ROOT / "configs/cnn_repetitions.json"),
        "source_sha256": sources, "cost_estimates": costs,
        "estimated_total_additional_seconds": sum(c["estimated_additional_seconds_with_margin"] for c in costs),
        "cost_method": "max measured full epoch per seed * 10 * 1.5 + 120s per export; estimate, not guarantee",
        "continuation": "last complete epoch, optimizer/RNG/patience/best model and history preserved",
        "selection": "best validation macro-F1 among all completed epochs, tie earlier epoch",
        "comparison": "paired per-seed changes versus ten epochs; mean and sample std; report all seeds",
        "decision": "report gain or no gain without seed selection; no significance claim with three seeds",
        "adaptive_origin": "extension chosen after inspecting ten-epoch validation curves",
        "test": "not_loaded_or_evaluated", "ensemble": False,
    }
    write_json(destination, plan)
    print(json.dumps({"plan": str(destination), "costs": costs,
                      "estimated_minutes": plan["estimated_total_additional_seconds"] / 60}, indent=2))


if __name__ == "__main__":
    main()
