"""Benchmark and extract block-six features from verified native log-Mel caches."""

import argparse
import csv
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from music_genre.data.audit import sha256_file, write_json
from music_genre.models.vit import FrozenViT
from music_genre.models.vit_block import FrozenViTBlock

ROOT = Path(__file__).resolve().parents[1]
WORKER = None


def initialize(config, mel_path):
    global WORKER
    torch.set_num_threads(config["threads_per_worker"])
    limiter = threadpool_limits(limits=config["threads_per_worker"])
    encoder = FrozenViTBlock(ROOT / config["checkpoint_directory"], config["feature_block"])
    WORKER = (config, encoder, np.load(mel_path, allow_pickle=False, mmap_mode="r"), limiter)


def extract_track(task):
    track_id, positions, cache_path, key = task
    config, encoder, mel, _ = WORKER
    target = Path(cache_path) / "tracks" / f"{track_id}.npz"
    metadata = target.with_suffix(".json")
    if target.exists() and metadata.exists():
        saved = json.loads(metadata.read_text())
        if saved["sha256"] != sha256_file(target):
            raise ValueError("Block-six track cache changed")
        with np.load(target, allow_pickle=False) as archive:
            if str(archive["cache_key"]) != key or not np.array_equal(archive["positions"], positions):
                raise ValueError("Stale block-six track cache")
            x = archive["x"]
    else:
        x = np.stack([encoder.extract(mel[i], config["db_min"], config["db_max"]) for i in positions])
        temporary = target.with_suffix(".partial.npz")
        np.savez(temporary, x=x, positions=positions, cache_key=key)
        temporary.replace(target)
        write_json(metadata, {"sha256": sha256_file(target)})
    if x.shape != (len(positions), 192) or not np.isfinite(x).all():
        raise ValueError("Invalid block-six vectors")
    return positions, x


def load_source(name):
    pointer = json.loads((ROOT / f"data/features/latest_{name}.json").read_text())
    folder = (ROOT / pointer["folder"]).resolve()
    if (not folder.is_relative_to((ROOT / "data/features").resolve())
            or sha256_file(folder / "report.json") != pointer["report_sha256"]):
        raise ValueError("Invalid source cache")
    report = json.loads((folder / "report.json").read_text())
    for filename, field in (("features.npy", "features_sha256"), ("index.csv", "index_sha256")):
        if sha256_file(folder / filename) != report[field]:
            raise ValueError("Source cache artifact changed")
    if report["test_audio_loaded"] or report["provenance"]["splits"] != ["train", "validation"]:
        raise ValueError("Source cache includes test")
    return pointer, folder, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be in 1..8")
    config = json.loads((ROOT / "configs/vit_block6.json").read_text())
    mp, mel_folder, mr = load_source("cnn_logmel")
    vp, vit_folder, vr = load_source("vit")
    if mr["index_sha256"] != vr["index_sha256"]:
        raise ValueError("Different source windows")
    for key in ("manifest_sha256", "labels"):
        if mr["provenance"][key] != vr["provenance"][key]:
            raise ValueError("Incompatible source metadata")
    for key in ("sample_rate", "window_seconds", "offsets_seconds", "logmel"):
        if config[key] != mr["provenance"]["config"][key]:
            raise ValueError("Cached log-Mel preprocessing differs")
    varying = {"name", "feature_block", "terminal_norm", "pooling"}
    if ({k: v for k, v in config.items() if k not in varying}
            != {k: v for k, v in vr["provenance"]["config"].items() if k not in varying}):
        raise ValueError("Unexpected difference from the block-twelve configuration")
    directory = (ROOT / config["checkpoint_directory"]).resolve()
    if not directory.is_relative_to((ROOT / "artifacts/pretrained").resolve()):
        raise ValueError("Checkpoint outside pretrained directory")
    for filename, digest in config["checkpoint"]["sha256"].items():
        if sha256_file(directory / filename) != digest:
            raise ValueError("Pinned checkpoint changed")
    with (mel_folder / "index.csv").open(encoding="utf-8", newline="") as stream:
        index = list(csv.DictReader(stream))
    groups, seen = {}, set()
    for i, row in enumerate(index):
        key = (row["track_id"], row["window_id"])
        if int(row["row_index"]) != i or row["split"] not in ("train", "validation") or key in seen:
            raise ValueError("Invalid source index")
        seen.add(key)
        groups.setdefault(row["track_id"], []).append(i)
    provenance = {
        "config": config, "parents": {"logmel": mp, "vit_reference": vp},
        "manifest_sha256": vr["provenance"]["manifest_sha256"], "labels": vr["provenance"]["labels"],
        "splits": ["train", "validation"],
        "versions": {name: version(name) for name in ("numpy", "torch", "timm", "safetensors")},
        "source_sha256": {name: sha256_file(ROOT / name) for name in (
            "scripts/extract_vit_block6.py", "src/music_genre/models/vit.py", "src/music_genre/models/vit_block.py",
        )},
    }
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    benchmark_path = ROOT / "reports/VIT_BLOCK6_BENCHMARK.json"
    mel_path = str(mel_folder / "features.npy")
    if args.benchmark:
        initialize(config, mel_path)
        _, encoder, mel, _ = WORKER
        full = FrozenViT(directory)
        reference = np.load(vit_folder / "features.npy", allow_pickle=False)
        rng = np.random.default_rng(42)
        selected = []
        for label in range(len(provenance["labels"])):
            group = [i for i, r in enumerate(index) if r["split"] == "train"
                     and int(r["label_index"]) == label and float(r["start_seconds"]) == 10.0]
            selected.extend(int(i) for i in rng.choice(group, 8, replace=False))
        timings, errors = [], []
        for number, position in enumerate([selected[0], *selected]):
            start = time.perf_counter()
            result = encoder.extract(mel[position], config["db_min"], config["db_max"])
            seconds = time.perf_counter() - start
            if result.shape != (192,):
                raise ValueError("Unexpected block-six dimension")
            check = full.extract(mel[position], config["db_min"], config["db_max"])
            errors.append(float(np.max(np.abs(check - reference[position]))))
            if number:
                timings.append(seconds)
        if max(errors) > 1e-6:
            raise ValueError("Cached log-Mel path does not reproduce the original block-twelve vectors")
        report = {"cache_key": key, "provenance": provenance, "sample_count": len(selected),
                  "sample_split": "train", "row_positions": selected,
                  "median_block6_seconds": float(np.median(timings)),
                  "block12_reference_max_abs_error": max(errors), "test_audio_loaded": False}
        write_json(benchmark_path, report)
        print(json.dumps({k: v for k, v in report.items() if k not in ("provenance", "row_positions")}, indent=2))
        return
    benchmark = json.loads(benchmark_path.read_text())
    if benchmark["cache_key"] != key:
        raise ValueError("Benchmark this exact extraction first")
    cache = ROOT / "data/features" / f"vit-block6-{key[:16]}"
    (cache / "tracks").mkdir(parents=True, exist_ok=True)
    if (cache / "report.json").exists():
        saved = json.loads((cache / "report.json").read_text())
        if saved["provenance"] != provenance:
            raise ValueError("Completed block-six provenance changed")
        for filename, field in (("features.npy", "features_sha256"), ("index.csv", "index_sha256")):
            if sha256_file(cache / filename) != saved[field]:
                raise ValueError("Completed block-six cache changed")
    else:
        began = time.perf_counter()
        matrix = np.empty((len(index), 192), dtype=np.float32)
        os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        print(f"Block 6: {len(groups)} tracks; test excluded; {cache}", flush=True)
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(config, mel_path)) as pool:
            tasks = ((track, positions, str(cache), key) for track, positions in groups.items())
            for count, (positions, values) in enumerate(pool.map(extract_track, tasks, chunksize=8), 1):
                matrix[positions] = values
                if count % 500 == 0:
                    print(f"Block 6: {count}/{len(groups)} tracks", flush=True)
        temporary = cache / "features.partial.npy"
        np.save(temporary, matrix, allow_pickle=False)
        temporary.replace(cache / "features.npy")
        (cache / "index.csv").write_bytes((mel_folder / "index.csv").read_bytes())
        saved = {
            "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(),
            "cache_key": key, "provenance": provenance, "tracks": len(groups),
            "matrix_shape": list(matrix.shape), "feature_names": [f"vit_block6_patch_mean_{i:03d}" for i in range(192)],
            "feature_families": ["vit"] * 192, "elapsed_seconds": time.perf_counter() - began,
            "features_sha256": sha256_file(cache / "features.npy"), "index_sha256": sha256_file(cache / "index.csv"),
            "benchmark_sha256": sha256_file(benchmark_path), "workers": args.workers,
            "audio_loaded": False, "test_audio_loaded": False,
        }
        write_json(cache / "report.json", saved)
    write_json(ROOT / "data/features/latest_vit_block6.json", {
        "folder": cache.relative_to(ROOT).as_posix(), "report_sha256": sha256_file(cache / "report.json"),
    })
    print(json.dumps({k: saved[k] for k in ("matrix_shape", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
