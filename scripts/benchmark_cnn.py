"""Benchmark 64 training windows and real optimizer steps before full CNN work."""

import csv
import json
import math
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from threadpoolctl import threadpool_limits

from music_genre.audio.processing import load_audio, segment_audio
from music_genre.data.audit import sha256_file, write_json
from music_genre.features.logmel import log_mel
from music_genre.models.cnn import SmallAudioCNN, normalize_logmel

ROOT = Path(__file__).resolve().parents[1]


def main():
    config = json.loads((ROOT / "configs/cnn.json").read_text())
    pointer = json.loads((ROOT / "data/manifests/latest_baseline.json").read_text())
    manifest = (ROOT / pointer["folder"] / "manifest.csv").resolve()
    if (not manifest.is_relative_to((ROOT / "data/manifests").resolve())
            or sha256_file(manifest) != pointer["manifest_sha256"]):
        raise ValueError("Baseline manifest changed or outside workspace")
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    labels = json.loads((ROOT / "configs/labels.json").read_text())["labels"]
    rng = np.random.default_rng(config["seed"])
    selection = []
    for label in labels:
        group = [r for r in rows if r["split"] == "train" and r["genre"] == label]
        selection.extend(group[int(i)] for i in rng.choice(len(group), 8, replace=False))
    matrices, targets, timings = [], [], []
    with threadpool_limits(limits=1):
        for i, row in enumerate([selection[0], *selection]):
            path = (ROOT / "data/raw" / row["audio_path"]).resolve()
            if (not path.is_relative_to((ROOT / "data/raw").resolve())
                    or sha256_file(path) != row["source_sha256"]):
                raise ValueError("Audio source changed")
            started = time.perf_counter()
            audio = load_audio(path, sample_rate=config["sample_rate"])
            window = segment_audio(audio, (10.0,), config["window_seconds"])[0]
            matrix = log_mel(window, config["logmel"])
            duration = time.perf_counter() - started
            if i:
                matrices.append(matrix)
                targets.append(int(row["label_index"]))
                timings.append(duration)
    torch.set_num_threads(config["threads"])
    torch.manual_seed(config["seed"])
    torch.use_deterministic_algorithms(True)
    model = SmallAudioCNN(len(labels)).to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                 weight_decay=config["weight_decay"])
    x = normalize_logmel(torch.from_numpy(np.stack(matrices))[:, None])
    y = torch.tensor(targets)
    batch = config["batch_size"]
    train_seconds, eval_seconds = [], []
    process = psutil.Process()
    peak_sampled_rss = process.memory_info().rss
    # One excluded warmup followed by eight full training batches over real train audio.
    with threadpool_limits(limits=config["threads"]):
        for step in range(9):
            start = (step % (len(x) // batch)) * batch
            xb = x[start:start + batch].contiguous(memory_format=torch.channels_last)
            yb = y[start:start + batch]
            began = time.perf_counter()
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(model(xb), yb)
            loss.backward()
            optimizer.step()
            if step:
                train_seconds.append(time.perf_counter() - began)
            peak_sampled_rss = max(peak_sampled_rss, process.memory_info().rss)
        model.eval()
        with torch.inference_mode():
            for start in range(0, len(x), batch):
                began = time.perf_counter()
                model(x[start:start + batch].contiguous(memory_format=torch.channels_last))
                eval_seconds.append(time.perf_counter() - began)
    train_windows = sum(r["split"] == "train" for r in rows) * 3
    val_windows = sum(r["split"] == "validation" for r in rows) * 3
    estimate = (math.ceil(train_windows / batch) * float(np.median(train_seconds))
                + math.ceil(val_windows / batch) * float(np.median(eval_seconds)))
    report = {
        "config": config, "config_sha256": sha256_file(ROOT / "configs/cnn.json"),
        "manifest_sha256": pointer["manifest_sha256"], "torch_version": str(torch.__version__),
        "sample_count": len(selection), "track_ids": [r["track_id"] for r in selection],
        "sample_split": "train", "spectrogram_shape": list(matrices[0].shape),
        "median_decode_logmel_seconds": float(np.median(timings)),
        "train_batch_seconds": train_seconds, "eval_batch_seconds": eval_seconds,
        "estimated_epoch_seconds": estimate,
        "estimated_max_epochs_seconds": estimate * config["max_epochs"],
        "estimate_excludes": ["disk_cache_reads", "checkpoint_writes", "metric_aggregation", "extraction"],
        "sampled_peak_rss_mib": peak_sampled_rss / 2**20,
        "parameters": sum(p.numel() for p in model.parameters()), "test_audio_loaded": False,
        "source_sha256": {name: sha256_file(ROOT / name) for name in (
            "scripts/benchmark_cnn.py", "src/music_genre/models/cnn.py",
            "src/music_genre/features/logmel.py", "src/music_genre/audio/processing.py",
        )},
    }
    write_json(ROOT / "reports/CNN_BENCHMARK.json", report)
    print(json.dumps({k: v for k, v in report.items() if k not in ("config", "track_ids", "source_sha256")}, indent=2))


if __name__ == "__main__":
    main()
