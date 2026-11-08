"""Run a train-only local diagnostic; does not train or evaluate a genre classifier."""

import csv
import hashlib
import json
import os
import platform
import statistics
import time
import tomllib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import psutil
import soundfile as sf
import torch

from music_genre.audio.processing import load_audio, segment_audio
from music_genre.diagnostics import environment_report
from music_genre.features.logmel import log_mel
from music_genre.models.vit import FrozenViT

ROOT = Path(__file__).resolve().parents[1]


def main():
    config_path = ROOT / "configs/diagnostic.toml"
    config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    labels = json.loads((ROOT / "configs/labels.json").read_text(encoding="utf-8"))["labels"]
    torch.set_num_threads(min(config["vit"]["threads"], os.cpu_count() or 1))
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    count = config["sample_count"]
    if count <= 0 or count % len(labels):
        raise ValueError("sample_count must be a positive multiple of the number of classes")
    groups = {label: [] for label in labels}
    with (ROOT / "data/raw/fma_metadata/tracks.csv").open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        parents, names = next(reader), next(reader)
        next(reader)
        columns = dict(zip(zip(parents, names), range(len(parents))))
        for row in reader:
            if (row[columns[("set", "subset")]] == "small"
                    and row[columns[("set", "split")]] == "training"):
                groups[row[columns[("track", "genre_top")]]].append(int(row[0]))
    rng = np.random.default_rng(config["seed"])
    selected = [(int(track_id), genre) for genre in labels
                for track_id in rng.choice(sorted(groups[genre]),
                                           size=count // len(labels), replace=False)]
    rng.shuffle(selected)
    start = time.perf_counter()
    extractor = FrozenViT(ROOT / config["vit"]["directory"])
    load_seconds = time.perf_counter() - start
    process = psutil.Process()
    rows, failures = [], []

    def run(track_id):
        name = f"{track_id:06d}"
        path = ROOT / "data/raw/fma_small" / name[:3] / f"{name}.mp3"
        start = time.perf_counter()
        audio = load_audio(path, config["audio"]["sample_rate"])
        windows = segment_audio(
            audio, (config["audio"]["start_seconds"],), config["audio"]["window_seconds"]
        )
        if not windows:
            raise ValueError("No audio at the requested diagnostic offset")
        decode_seconds = time.perf_counter() - start
        start = time.perf_counter()
        mel = log_mel(windows[0], config["logmel"])
        mel_seconds = time.perf_counter() - start
        start = time.perf_counter()
        embedding = extractor.extract(mel, config["vit"]["db_min"], config["vit"]["db_max"])
        vit_seconds = time.perf_counter() - start
        return {
            "track_id": name, "audio_path": path.relative_to(ROOT).as_posix(),
            "source_sample_rate": audio.source_sample_rate,
            "source_channels": audio.source_channels,
            "start_seconds": windows[0].start_seconds,
            "valid_seconds": windows[0].valid_samples / audio.sample_rate,
            "logmel_shape": list(mel.shape), "embedding_shape": list(embedding.shape),
            "embedding_finite": bool(np.isfinite(embedding).all()),
            "db_clipped_fraction": float(np.mean(
                (mel < config["vit"]["db_min"]) | (mel > config["vit"]["db_max"])
            )),
            "decode_resample_seconds": decode_seconds, "logmel_seconds": mel_seconds,
            "vit_seconds": vit_seconds,
            "total_seconds": decode_seconds + mel_seconds + vit_seconds,
        }

    print("Warm-up on a training example (includes first-call compilation)...", flush=True)
    start = time.perf_counter()
    run(selected[0][0])
    warmup_seconds = time.perf_counter() - start
    print(f"Warm-up complete: {warmup_seconds:.2f}s", flush=True)
    for position, (track_id, genre) in enumerate(selected, start=1):
        try:
            row = run(track_id)
            row["genre"] = genre
            rows.append(row)
        except (ValueError, RuntimeError, OSError) as error:
            failures.append({"track_id": track_id, "genre": genre, "error": str(error)})
        if position % 8 == 0:
            print(f"Diagnostic: {position}/{count} tracks processed", flush=True)
    memory = process.memory_info()
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(),
        "status": "passed" if len(rows) == count and not failures else "partial_failure",
        "purpose": "feasibility_only_no_genre_accuracy",
        "selection": {"split": "training", "seed": config["seed"], "requested": count,
                      "successful": len(rows), "genres": dict(Counter(r["genre"] for r in rows))},
        "environment": environment_report(ROOT), "host_architecture_reported": platform.machine(),
        "torch_cuda_available": torch.cuda.is_available(),
        "torch_threads": torch.get_num_threads(), "libsndfile": sf.__libsndfile_version__,
        "physical_ram_gib": round(psutil.virtual_memory().total / 2**30, 2),
        "process_rss_mib": round(memory.rss / 2**20, 2),
        "process_peak_working_set_mib": round(getattr(memory, "peak_wset", memory.rss) / 2**20, 2),
        "model_load_seconds": load_seconds, "first_call_warmup_seconds": warmup_seconds,
        "config": config, "checkpoint": extractor.provenance,
        "source_sha256": {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted((ROOT / "src/music_genre").rglob("*.py"))},
        "failures": failures, "samples": rows,
        "limitations": [
            "64 training excerpts; not a full dataset decode audit",
            "warm timings exclude imports, model load and first-call compilation",
            "fixed dB mapping for diagnosis only; no claim of optimal audio representation",
            "CPU inference only; no estimate of fine-tuning cost",
        ],
    }
    if rows:
        report["warm_median_seconds"] = {
            key: statistics.median(row[key] for row in rows)
            for key in ("decode_resample_seconds", "logmel_seconds", "vit_seconds", "total_seconds")
        }
        report["warm_p95_total_seconds"] = float(np.percentile(
            [r["total_seconds"] for r in rows], 95
        ))
    output = ROOT / "reports/environment_benchmark.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("status", "selection", "warm_median_seconds")
                      if k in report}, indent=2), flush=True)
    print(f"Report: {output}", flush=True)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
