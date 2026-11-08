"""Train a bounded CPU CNN; select epochs by track-level validation macro-F1."""

import argparse
import csv
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file, save_file
from threadpoolctl import threadpool_limits

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.metrics import aggregate_tracks, classification_metrics
from music_genre.models.cnn import SmallAudioCNN
from music_genre.training.cnn import (
    epoch_order,
    load_checkpoint,
    make_batch,
    predict_windows,
    save_checkpoint,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config_path = ROOT / "configs/cnn.json"
    config = json.loads(config_path.read_text())
    benchmark_path = ROOT / "reports/CNN_BENCHMARK.json"
    benchmark = json.loads(benchmark_path.read_text())
    if benchmark["config_sha256"] != sha256_file(config_path):
        raise ValueError("Benchmark this CNN configuration first")
    if benchmark["estimated_max_epochs_seconds"] > config["budget_seconds"]:
        raise ValueError("Benchmark predicts an experiment beyond its time budget")
    pointer = json.loads((ROOT / "data/features/latest_cnn_logmel.json").read_text())
    cache = (ROOT / pointer["folder"]).resolve()
    if (not cache.is_relative_to((ROOT / "data/features").resolve())
            or sha256_file(cache / "report.json") != pointer["report_sha256"]):
        raise ValueError("Invalid CNN feature cache")
    features = json.loads((cache / "report.json").read_text())
    for filename, key in (("features.npy", "features_sha256"), ("index.csv", "index_sha256")):
        if sha256_file(cache / filename) != features[key]:
            raise ValueError("CNN cache artifact changed")
    baseline = json.loads((ROOT / "data/manifests/latest_baseline.json").read_text())
    manifest = (ROOT / baseline["folder"] / "manifest.csv").resolve()
    if (not manifest.is_relative_to((ROOT / "data/manifests").resolve())
            or sha256_file(manifest) != baseline["manifest_sha256"]
            or features["provenance"]["manifest_sha256"] != baseline["manifest_sha256"]):
        raise ValueError("Baseline changed")
    labels = json.loads((ROOT / "configs/labels.json").read_text())["labels"]
    if features["provenance"]["labels"] != labels:
        raise ValueError("Class order changed")
    with manifest.open(encoding="utf-8", newline="") as stream:
        sources = {r["track_id"]: r for r in csv.DictReader(stream)
                   if r["split"] in ("train", "validation")}
    with (cache / "index.csv").open(encoding="utf-8", newline="") as stream:
        index = list(csv.DictReader(stream))
    seen = set()
    for i, row in enumerate(index):
        source = sources[row["track_id"]]
        key = (row["track_id"], int(row["window_id"]))
        if (int(row["row_index"]) != i or key in seen
                or any(row[k] != source[k] for k in ("split", "artist_id", "label_index", "source_sha256"))):
            raise ValueError("Invalid window index")
        seen.add(key)
    if {r["track_id"] for r in index} != set(sources):
        raise ValueError("Missing reference tracks")
    positions = {split: np.array([i for i, r in enumerate(index) if r["split"] == split])
                 for split in ("train", "validation")}
    train_index = [index[i] for i in positions["train"]]
    val_index = [index[i] for i in positions["validation"]]
    if {r["artist_id"] for r in train_index} & {r["artist_id"] for r in val_index}:
        raise ValueError("Artist overlap")
    matrix = np.load(cache / "features.npy", allow_pickle=False, mmap_mode="r")
    if matrix.shape != (len(index), 128, 622) or matrix.dtype != np.float32:
        raise ValueError("Unexpected native log-Mel matrix")
    y = np.array([int(r["label_index"]) for r in index], dtype=np.int64)
    if set(y[positions["train"]]) != set(range(len(labels))):
        raise ValueError("Missing training class")
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(config["deterministic"])
    identity = {
        "config": config, "cache": pointer, "torch_version": str(torch.__version__),
        "numpy_version": np.__version__, "benchmark_sha256": sha256_file(benchmark_path),
        "source_sha256": {name: sha256_file(ROOT / name) for name in (
            "scripts/train_cnn.py", "src/music_genre/training/cnn.py",
            "src/music_genre/models/cnn.py", "src/music_genre/evaluation/metrics.py",
        )},
    }
    model = SmallAudioCNN(len(labels)).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    if args.resume:
        folder = args.resume.resolve()
        if not folder.is_relative_to((ROOT / "runs").resolve()):
            raise ValueError("Resume directory outside runs")
        state = load_checkpoint(folder / "checkpoint.pt", identity)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["rng"])
    else:
        folder = ROOT / "runs" / ("cnn-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
        folder.mkdir(parents=True, exist_ok=False)
        write_json(folder / "config.json", config)
        write_json(folder / "feature_config.json", features["provenance"]["config"])
        state = {"identity": identity, "epoch": 0, "history": [], "best_epoch": None,
                 "best_score": -1.0, "best_model": None, "bad_epochs": 0, "elapsed_seconds": 0.0,
                 "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": torch.get_rng_state()}
        save_checkpoint(folder / "checkpoint.pt", state)
    write_json(ROOT / "reports/latest_cnn_training.json", {"folder": folder.relative_to(ROOT).as_posix()})
    print(f"Run: {folder}; resume from epoch {state['epoch']}; test excluded", flush=True)
    began = time.perf_counter()
    previous_elapsed = state["elapsed_seconds"]
    stop_reason = "max_epochs"
    with threadpool_limits(limits=config["threads"]):
        for epoch in range(state["epoch"] + 1, config["max_epochs"] + 1):
            if state["bad_epochs"] >= config["patience"]:
                stop_reason = "early_stopping"
                break
            epoch_start = time.perf_counter()
            model.train()
            order = epoch_order(positions["train"], config["seed"], epoch)
            loss_sum = 0.0
            budget_reached = False
            for start in range(0, len(order), config["batch_size"]):
                if previous_elapsed + time.perf_counter() - began >= config["budget_seconds"]:
                    budget_reached = True
                    break
                batch_positions = order[start:start + config["batch_size"]]
                xb = make_batch(matrix, batch_positions, config)
                targets = torch.from_numpy(y[batch_positions])
                optimizer.zero_grad(set_to_none=True)
                loss = torch.nn.functional.cross_entropy(model(xb), targets)
                if not torch.isfinite(loss):
                    raise ValueError("Non-finite training loss")
                loss.backward()
                optimizer.step()
                loss_sum += float(loss.detach()) * len(batch_positions)
                if start and (start // config["batch_size"]) % 400 == 0:
                    print(f"Epoch {epoch}: {start}/{len(order)} training windows", flush=True)
            if budget_reached:
                stop_reason = "budget_exhausted_incomplete_epoch_discarded"
                # Restore the last complete epoch and retain spent compute in the budget.
                state = load_checkpoint(folder / "checkpoint.pt", identity)
                state["elapsed_seconds"] = previous_elapsed + time.perf_counter() - began
                save_checkpoint(folder / "checkpoint.pt", state)
                break
            probabilities = predict_windows(model, matrix, positions["validation"], config)
            _, truth, track_probabilities = aggregate_tracks(val_index, probabilities, len(labels))
            metrics = classification_metrics(truth, track_probabilities, labels)
            if metrics["macro_f1"] > state["best_score"]:
                state["best_score"], state["best_epoch"] = metrics["macro_f1"], epoch
                state["best_model"] = {k: v.detach().clone().contiguous() for k, v in model.state_dict().items()}
                state["bad_epochs"] = 0
            else:
                state["bad_epochs"] += 1
            state["history"].append({"epoch": epoch, "training_window_loss": loss_sum / len(order),
                                     "validation": metrics, "seconds": time.perf_counter() - epoch_start})
            state.update(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         rng=torch.get_rng_state(), elapsed_seconds=previous_elapsed + time.perf_counter() - began)
            save_checkpoint(folder / "checkpoint.pt", state)
            write_json(folder / "progress.json", {
                "status": "training", "epoch": epoch, "best_epoch": state["best_epoch"],
                "best_validation_macro_f1": state["best_score"], "history": state["history"],
                "elapsed_seconds": state["elapsed_seconds"], "test_evaluated": False,
            })
            print(f"Epoch {epoch}: loss={loss_sum / len(order):.4f}, "
                  f"validation macro-F1={metrics['macro_f1']:.4f}, accuracy={metrics['accuracy']:.4f}, "
                  f"seconds={time.perf_counter() - epoch_start:.1f}", flush=True)
        elapsed = previous_elapsed + time.perf_counter() - began
        if state["best_model"] is None:
            write_json(folder / "progress.json", {"status": "budget_exhausted", "elapsed_seconds": elapsed})
            raise RuntimeError("No complete epoch within budget; no result published")
        model.load_state_dict(state["best_model"])
        window_probabilities = predict_windows(model, matrix, positions["validation"], config)
        track_ids, truth, probabilities = aggregate_tracks(val_index, window_probabilities, len(labels))
        validation = classification_metrics(truth, probabilities, labels)
        if validation["macro_f1"] != state["best_score"]:
            raise ValueError("Restored best epoch does not reproduce selection score")
        save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()}, str(folder / "model.safetensors"))
        restored = SmallAudioCNN(len(labels)).to(memory_format=torch.channels_last)
        restored.load_state_dict(load_file(str(folder / "model.safetensors")), strict=True)
        restored_probabilities = predict_windows(restored, matrix, positions["validation"], config)
        export_error = float(np.max(np.abs(restored_probabilities - window_probabilities)))
        if export_error > 1e-6:
            raise ValueError("CNN export changed predictions")
        train_ids, train_truth, train_probabilities = aggregate_tracks(
            train_index, predict_windows(model, matrix, positions["train"], config), len(labels)
        )
    prediction_rows = [
        {"track_id": key, "true_label": labels[int(target)], "predicted_label": labels[int(p.argmax())],
         **{f"p_{label}": float(p[i]) for i, label in enumerate(labels)}}
        for key, target, p in zip(track_ids, truth, probabilities, strict=True)
    ]
    with (folder / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    report = {
        "schema_version": 1, "config": config, "identity": identity, "labels": labels,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "budget_exhausted" if stop_reason.startswith("budget") else "complete", "stop_reason": stop_reason,
        "feature_cache": pointer["folder"], "feature_report_sha256": pointer["report_sha256"],
        "baseline_manifest_sha256": baseline["manifest_sha256"], "index_sha256": features["index_sha256"],
        "track_counts": {"train": len(train_ids), "validation": len(track_ids)},
        "window_counts": {k: len(v) for k, v in positions.items()}, "history": state["history"],
        "selected_epoch": state["best_epoch"], "epochs_completed": state["epoch"],
        "parameters": sum(p.numel() for p in model.parameters()),
        "validation": validation, "training": classification_metrics(train_truth, train_probabilities, labels),
        "training_validation_seconds": elapsed,
        "elapsed_including_export_and_train_evaluation_seconds": previous_elapsed + time.perf_counter() - began,
        "export_max_probability_error": export_error,
        "model_sha256": sha256_file(folder / "model.safetensors"),
        "validation_predictions_sha256": sha256_file(folder / "validation_predictions.csv"),
        "test_loaded": False, "test_evaluated": False,
        "limitations": ["One development seed; final stochastic repetitions pending.",
                        "Epoch selection uses validation; no independent final estimate.",
                        "Native log-Mel CNN from scratch versus resized pretrained image ViT: not an architecture-only ablation."],
    }
    write_json(folder / "model.json", {
        "architecture": "SmallAudioCNN", "config": config, "labels": labels,
        "model_sha256": report["model_sha256"], "aggregation": "mean_window_probabilities",
        "selected_epoch": state["best_epoch"], "test_evaluated": False,
    })
    write_json(folder / "report.json", report)
    write_json(folder / "progress.json", {"status": report["status"], "selected_epoch": state["best_epoch"],
                                         "stop_reason": stop_reason, "validation": validation})
    write_json(ROOT / "reports/latest_cnn_run.json", {"folder": folder.relative_to(ROOT).as_posix(),
               "report_sha256": sha256_file(folder / "report.json")})
    print(f"Selected epoch {state['best_epoch']}: macro-F1={validation['macro_f1']:.4f}; {folder}", flush=True)


if __name__ == "__main__":
    main()
