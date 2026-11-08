"""Benchmark then extract training/validation vit, using a provenance-keyed cache."""

import argparse
import csv
import hashlib
import json
import os
import platform
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from music_genre.audio.processing import load_audio, segment_audio
from music_genre.data.audit import sha256_file, write_json
from music_genre.features.hpss import MusicalHpssViT
from music_genre.features.logmel import log_mel

ROOT = Path(__file__).resolve().parents[1]
EXTRACTOR = None


def reference_cache():
    pointer = json.loads((ROOT / "data/features/latest_vit.json").read_text(encoding="utf-8"))
    folder = (ROOT / pointer["folder"]).resolve()
    if not folder.is_relative_to((ROOT / "data/features").resolve()):
        raise ValueError("Reference outside feature directory")
    if sha256_file(folder / "report.json") != pointer["report_sha256"]:
        raise ValueError("Reference report changed")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    for filename, key in (("features.npy", "features_sha256"), ("index.csv", "index_sha256")):
        if sha256_file(folder / filename) != report[key]:
            raise ValueError("Reference artifacts changed")
    with (folder / "index.csv").open(encoding="utf-8", newline="") as stream:
        index = list(csv.DictReader(stream))
    matrix = np.load(folder / "features.npy", allow_pickle=False)
    if matrix.shape != (len(index), 192) or not np.isfinite(matrix).all():
        raise ValueError("Invalid reference matrix")
    return pointer, report, index, matrix


def initialize(config):
    global EXTRACTOR
    # Worker lifetime: retain the limiter so numerical libraries use one thread.
    EXTRACTOR = (MusicalHpssViT(config, ROOT), threadpool_limits(limits=1))


def extract_track(task):
    row, cache_string, key = task
    cache = Path(cache_string)
    extractor = EXTRACTOR[0]
    config = extractor.config
    path = (ROOT / "data/raw" / row["audio_path"]).resolve()
    if not path.is_relative_to((ROOT / "data/raw").resolve()):
        raise ValueError("Audio outside raw directory")
    if sha256_file(path) != row["source_sha256"]:
        raise ValueError(f"Source changed: {row['track_id']}")
    target = cache / "tracks" / f"{row['track_id']}.npz"
    checksum = target.with_suffix(".json")
    if target.exists() and checksum.exists():
        saved_hash = json.loads(checksum.read_text(encoding="utf-8"))
        if saved_hash["sha256"] != sha256_file(target):
            raise ValueError("Track cache checksum mismatch")
        with np.load(target, allow_pickle=False) as saved:
            if str(saved["cache_key"]) != key or str(saved["source_sha256"]) != row["source_sha256"]:
                raise ValueError("Stale track cache")
            x, starts, valid = saved["x"], saved["starts"], saved["valid"]
    else:
        audio = load_audio(path, sample_rate=config["sample_rate"])
        windows = segment_audio(audio, tuple(config["offsets_seconds"]), config["window_seconds"])
        if not windows:
            raise ValueError("No audio windows")
        x = np.stack([extractor.extract(window) for window in windows])
        starts = np.array([w.start_seconds for w in windows], dtype=np.float64)
        valid = np.array([w.valid_samples / w.sample_rate for w in windows], dtype=np.float64)
        temporary = target.with_suffix(".partial.npz")
        np.savez(temporary, x=x, starts=starts, valid=valid,
                 cache_key=key, source_sha256=row["source_sha256"])
        temporary.replace(target)
        write_json(checksum, {"sha256": sha256_file(target)})
    if (x.shape != (len(starts), len(extractor.names)) or not len(starts)
            or len(starts) != len(valid) or not np.isfinite(x).all()
            or not np.isfinite(starts).all() or not np.isfinite(valid).all()
            or not np.all(valid > 0) or len(set(starts)) != len(starts)):
        raise ValueError("Invalid cached windows")
    return row, x, starts, valid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be in 1..8")
    config = json.loads((ROOT / "configs/vit_hpss.json").read_text(encoding="utf-8"))
    pointer = json.loads((ROOT / "data/manifests/latest_baseline.json").read_text(encoding="utf-8"))
    manifest = (ROOT / pointer["folder"] / "manifest.csv").resolve()
    if not manifest.is_relative_to((ROOT / "data/manifests").resolve()):
        raise ValueError("Manifest outside data directory")
    if sha256_file(manifest) != pointer["manifest_sha256"]:
        raise ValueError("Baseline manifest changed")
    with manifest.open(encoding="utf-8", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    rows = [row for row in all_rows if row["split"] in ("train", "validation")]
    if any(r["status"] != "valid" or r["decision"] != "keep" for r in rows):
        raise ValueError("Unexpected non-selected track")
    labels = json.loads((ROOT / "configs/labels.json").read_text(encoding="utf-8"))["labels"]
    if any(int(r["label_index"]) != labels.index(r["genre"]) for r in rows):
        raise ValueError("Label mapping mismatch")
    parent, reference, ref_index, ref_matrix = reference_cache()
    allowed = {"name", "channels", "hpss"}
    if ({k: v for k, v in config.items() if k not in allowed}
            != {k: v for k, v in reference["provenance"]["config"].items() if k not in allowed}
            or reference["provenance"]["manifest_sha256"] != pointer["manifest_sha256"]
            or reference["provenance"]["labels"] != labels):
        raise ValueError("HPSS must change only the channels of the block-12 reference")
    provenance = {
        "config": config, "manifest_sha256": pointer["manifest_sha256"],
        "labels": labels, "splits": ["train", "validation"],
        "parents": {"vit_reference": parent},
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("numpy", "scipy", "librosa", "soundfile", "torch", "timm", "safetensors")},
        "source_sha256": {name: sha256_file(ROOT / name) for name in (
            "src/music_genre/features/hpss.py", "src/music_genre/models/vit_channels.py", "src/music_genre/audio/processing.py",
            "src/music_genre/features/logmel.py", "src/music_genre/models/vit.py",
            "scripts/extract_vit_hpss.py",
        )},
    }
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    extractor = MusicalHpssViT(config, ROOT)
    benchmark_path = ROOT / "reports/VIT_HPSS_BENCHMARK.json"
    if args.benchmark:
        rng = np.random.default_rng(42)
        selection = []
        for label in labels:
            group = [r for r in rows if r["split"] == "train" and r["genre"] == label]
            selection.extend(group[int(i)] for i in rng.choice(len(group), 8, replace=False))
        samples = []
        reference_positions = {r["track_id"]: i for i, r in enumerate(ref_index)
                               if r["split"] == "train" and float(r["start_seconds"]) == 10.0}
        mel_errors, embedding_errors = [], []
        with threadpool_limits(limits=1):
            for i, row in enumerate([selection[0], *selection]):
                path = ROOT / "data/raw" / row["audio_path"]
                if sha256_file(path) != row["source_sha256"]:
                    raise ValueError("Benchmark source changed")
                started = time.perf_counter()
                audio = load_audio(path, sample_rate=config["sample_rate"])
                window = segment_audio(audio, (10.0,), config["window_seconds"])[0]
                decoded = time.perf_counter()
                channels = extractor.spectrograms(window)
                x = extractor.encoder.extract(channels, config["db_min"], config["db_max"])
                finished = time.perf_counter()
                original = log_mel(window, config["logmel"])
                mel_error = float(np.max(np.abs(channels[0] - original)))
                repeated = extractor.encoder.extract(
                    np.repeat(channels[:1], 3, axis=0), config["db_min"], config["db_max"]
                )
                embedding_error = float(np.max(np.abs(
                    repeated - ref_matrix[reference_positions[row["track_id"]]]
                )))
                if mel_error > 1e-5 or embedding_error > 1e-6:
                    raise ValueError("Original channel does not reproduce the block-12 reference")
                if i:
                    mel_errors.append(mel_error)
                    embedding_errors.append(embedding_error)
                    samples.append({"track_id": row["track_id"], "genre": row["genre"],
                                    "decode_seconds": decoded - started,
                                    "features_seconds": finished - decoded,
                                    "dimensions": len(x)})
                else:
                    warmup_seconds = finished - started
        report = {
            "cache_key": key, "provenance": provenance, "sample_count": len(samples),
            "split": "train", "warmup_seconds": warmup_seconds,
            "median_decode_seconds": float(np.median([r["decode_seconds"] for r in samples])),
            "median_features_seconds": float(np.median([r["features_seconds"] for r in samples])),
            "original_logmel_max_abs_error": max(mel_errors),
            "repeated_original_embedding_max_abs_error": max(embedding_errors),
            "samples": samples, "test_audio_loaded": False,
        }
        write_json(benchmark_path, report)
        print(json.dumps({k: v for k, v in report.items() if k not in ("samples", "provenance")}, indent=2))
        return
    if not benchmark_path.exists() or json.loads(benchmark_path.read_text(encoding="utf-8"))["cache_key"] != key:
        raise ValueError("Run --benchmark for this exact configuration before extraction")
    cache = ROOT / "data/features" / f"vit-hpss-{key[:16]}"
    (cache / "tracks").mkdir(parents=True, exist_ok=True)
    if (cache / "report.json").exists():
        complete = json.loads((cache / "report.json").read_text(encoding="utf-8"))
        if (complete["cache_key"] != key
                or complete["features_sha256"] != sha256_file(cache / "features.npy")
                or complete["index_sha256"] != sha256_file(cache / "index.csv")
                or complete["index_sha256"] != reference["index_sha256"]):
            raise ValueError("Completed cache changed")
        write_json(ROOT / "data/features/latest_vit_hpss.json",
                   {"folder": cache.relative_to(ROOT).as_posix(),
                    "report_sha256": sha256_file(cache / "report.json")})
        print("Verified completed HPSS cache; preserving original extraction report")
        return
    matrices, index = [], []
    started = time.perf_counter()
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    print(f"Cache: {cache}; tracks: {len(rows)}; test excluded", flush=True)
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize,
                                 initargs=(config,)) as pool:
            for count, (row, x, starts, valid) in enumerate(
                pool.map(extract_track, ((r, str(cache), key) for r in rows), chunksize=8), start=1
            ):
                matrices.append(x)
                for window_id, (start, duration) in enumerate(zip(starts, valid, strict=True)):
                    index.append({
                        "row_index": len(index), "track_id": row["track_id"], "window_id": window_id,
                        "start_seconds": float(start), "valid_seconds": float(duration),
                        "padded_seconds": max(0, config["window_seconds"] - float(duration)),
                        "split": row["split"], "label_index": row["label_index"],
                        "artist_id": row["artist_id"], "source_sha256": row["source_sha256"],
                    })
                if count % 250 == 0:
                    print(f"ViT HPSS: {count}/{len(rows)} tracks", flush=True)
    except (ValueError, RuntimeError, OSError) as error:
        write_json(cache / "failure.json", {"error": str(error), "cache_key": key})
        raise
    matrix = np.concatenate(matrices)
    np.save(cache / "features.npy", matrix, allow_pickle=False)
    with (cache / "index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    if sha256_file(cache / "index.csv") != reference["index_sha256"]:
        raise ValueError("HPSS and reference windows differ")
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(),
        "cache_key": key, "provenance": provenance, "tracks": len(rows),
        "matrix_shape": list(matrix.shape), "feature_names": extractor.names,
        "feature_families": extractor.families, "elapsed_seconds": time.perf_counter() - started,
        "workers": args.workers, "benchmark_sha256": sha256_file(benchmark_path),
        "features_sha256": sha256_file(cache / "features.npy"),
        "index_sha256": sha256_file(cache / "index.csv"), "test_audio_loaded": False,
    }
    write_json(cache / "report.json", report)
    write_json(ROOT / "data/features/latest_vit_hpss.json",
               {"folder": cache.relative_to(ROOT).as_posix(),
                "report_sha256": sha256_file(cache / "report.json")})
    print(json.dumps({k: report[k] for k in ("tracks", "matrix_shape", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
