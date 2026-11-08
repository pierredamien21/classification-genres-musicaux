"""Resume per-track log-Mel extraction, then assemble an on-disk float32 array."""

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from music_genre.audio.processing import load_audio, segment_audio
from music_genre.data.audit import sha256_file, write_json
from music_genre.features.logmel import log_mel

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = None
LIMITER = None


def initialize(config):
    global SETTINGS, LIMITER
    SETTINGS = config
    LIMITER = threadpool_limits(limits=1)


def extract_track(task):
    row, expected, folder = task
    cache = Path(folder)
    path = (ROOT / "data/raw" / row["audio_path"]).resolve()
    if (not path.is_relative_to((ROOT / "data/raw").resolve())
            or sha256_file(path) != row["source_sha256"]):
        raise ValueError("Audio path or source changed")
    target = cache / "tracks" / f"{row['track_id']}.npy"
    metadata_path = target.with_suffix(".json")
    if target.exists() and metadata_path.exists():
        saved = json.loads(metadata_path.read_text())
        if (saved["source_sha256"] != row["source_sha256"] or saved["index"] != expected
                or sha256_file(target) != saved["sha256"]):
            raise ValueError("Track cache changed")
        matrix = np.load(target, allow_pickle=False, mmap_mode="r")
    else:
        audio = load_audio(path, sample_rate=SETTINGS["sample_rate"])
        windows = segment_audio(audio, tuple(SETTINGS["offsets_seconds"]), SETTINGS["window_seconds"])
        if len(windows) != len(expected):
            raise ValueError("Window count differs from reference")
        for window, reference in zip(windows, expected, strict=True):
            if (window.start_seconds != float(reference["start_seconds"])
                    or window.valid_samples / window.sample_rate != float(reference["valid_seconds"])):
                raise ValueError("Window timing differs from reference")
        matrix = np.stack([log_mel(w, SETTINGS["logmel"]) for w in windows])
        temporary = target.with_suffix(".partial.npy")
        np.save(temporary, matrix, allow_pickle=False)
        temporary.replace(target)
        saved = {"source_sha256": row["source_sha256"], "index": expected, "sha256": sha256_file(target)}
        write_json(metadata_path, saved)
    if matrix.shape != (len(expected), 128, 622) or matrix.dtype != np.float32 or not np.isfinite(matrix).all():
        raise ValueError("Invalid cached spectrograms")
    return row["track_id"], saved["sha256"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be in 1..8")
    config = json.loads((ROOT / "configs/cnn.json").read_text())
    benchmark = json.loads((ROOT / "reports/CNN_BENCHMARK.json").read_text())
    if benchmark["config_sha256"] != sha256_file(ROOT / "configs/cnn.json"):
        raise ValueError("Benchmark this configuration first")
    for name, checksum in benchmark["source_sha256"].items():
        if sha256_file(ROOT / name) != checksum:
            raise ValueError("Benchmark source changed")
    baseline = json.loads((ROOT / "data/manifests/latest_baseline.json").read_text())
    manifest = (ROOT / baseline["folder"] / "manifest.csv").resolve()
    if (not manifest.is_relative_to((ROOT / "data/manifests").resolve())
            or sha256_file(manifest) != baseline["manifest_sha256"]
            or benchmark["manifest_sha256"] != baseline["manifest_sha256"]):
        raise ValueError("Manifest changed")
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] in ("train", "validation")]
    pointer = json.loads((ROOT / "data/features/latest_descriptors.json").read_text())
    reference = (ROOT / pointer["folder"]).resolve()
    if (not reference.is_relative_to((ROOT / "data/features").resolve())
            or sha256_file(reference / "report.json") != pointer["report_sha256"]):
        raise ValueError("Reference report changed")
    report = json.loads((reference / "report.json").read_text())
    if (report["provenance"]["manifest_sha256"] != baseline["manifest_sha256"]
            or sha256_file(reference / "index.csv") != report["index_sha256"]):
        raise ValueError("Reference selection changed")
    with (reference / "index.csv").open(encoding="utf-8", newline="") as stream:
        index = list(csv.DictReader(stream))
    groups = {}
    for i, item in enumerate(index):
        if item["split"] not in ("train", "validation") or int(item["row_index"]) != i:
            raise ValueError("Unexpected reference index")
        groups.setdefault(item["track_id"], []).append(item)
    if set(groups) != {r["track_id"] for r in rows}:
        raise ValueError("Reference tracks differ")
    for row in rows:
        if row["decision"] != "keep" or row["status"] != "valid":
            raise ValueError("Unexpected unselected source")
        for item in groups[row["track_id"]]:
            if any(item[k] != row[k] for k in ("source_sha256", "artist_id", "split", "label_index")):
                raise ValueError("Reference index differs from manifest")
    audio_config = {k: config[k] for k in ("sample_rate", "window_seconds", "offsets_seconds", "logmel")}
    provenance = {
        "config": audio_config, "manifest_sha256": baseline["manifest_sha256"],
        "reference": pointer, "labels": report["provenance"]["labels"], "splits": ["train", "validation"],
        "versions": {name: version(name) for name in ("numpy", "librosa", "soundfile", "scipy")},
        "source_sha256": {name: sha256_file(ROOT / name) for name in (
            "scripts/extract_cnn_logmel.py", "src/music_genre/features/logmel.py",
            "src/music_genre/audio/processing.py",
        )},
    }
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    cache = ROOT / "data/features" / f"cnn-logmel-{key[:16]}"
    (cache / "tracks").mkdir(parents=True, exist_ok=True)
    if (cache / "report.json").exists():
        saved = json.loads((cache / "report.json").read_text())
        if saved["provenance"] != provenance:
            raise ValueError("Cache provenance changed")
        for filename, checksum in (("features.npy", "features_sha256"), ("index.csv", "index_sha256")):
            if sha256_file(cache / filename) != saved[checksum]:
                raise ValueError("Assembled cache changed")
        write_json(ROOT / "data/features/latest_cnn_logmel.json", {
            "folder": cache.relative_to(ROOT).as_posix(), "report_sha256": sha256_file(cache / "report.json"),
        })
        print("Verified and reused complete CNN log-Mel cache")
        return
    required = len(index) * 128 * 622 * 4 * 2
    if shutil.disk_usage(cache).free < required + 2**30:
        raise ValueError("Insufficient free disk for resumable spectrogram cache")
    began = time.perf_counter()
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    print(f"Extracting {len(rows)} train/validation tracks; cache: {cache}", flush=True)
    checksums = {}
    with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(audio_config,)) as pool:
        tasks = ((r, groups[r["track_id"]], str(cache)) for r in rows)
        for count, (track_id, checksum) in enumerate(pool.map(extract_track, tasks, chunksize=4), 1):
            checksums[track_id] = checksum
            if count % 500 == 0:
                print(f"Log-Mel: {count}/{len(rows)} tracks", flush=True)
    temporary = cache / "features.partial.npy"
    matrix = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.float32, shape=(len(index), 128, 622))
    for row in rows:
        positions = [int(r["row_index"]) for r in groups[row["track_id"]]]
        matrix[positions] = np.load(cache / "tracks" / f"{row['track_id']}.npy", allow_pickle=False)
    matrix.flush()
    del matrix
    temporary.replace(cache / "features.npy")
    (cache / "index.csv").write_bytes((reference / "index.csv").read_bytes())
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(),
        "provenance": provenance, "cache_key": key, "tracks": len(rows),
        "matrix_shape": [len(index), 128, 622], "dtype": "float32", "track_sha256": checksums,
        "elapsed_seconds": time.perf_counter() - began,
        "features_sha256": sha256_file(cache / "features.npy"),
        "index_sha256": sha256_file(cache / "index.csv"), "test_audio_loaded": False,
    }
    write_json(cache / "report.json", report)
    write_json(ROOT / "data/features/latest_cnn_logmel.json", {
        "folder": cache.relative_to(ROOT).as_posix(), "report_sha256": sha256_file(cache / "report.json"),
    })
    print(json.dumps({k: report[k] for k in ("matrix_shape", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
