"""Extract one cached representation (any ViT variant or musical descriptors) for train/validation.

Replaces the five V1 scripts extract_vit*.py and extract_descriptors.py (kept at tag v1-final).
The representation is read from --config: a config with `checkpoint_directory` is a frozen ViT
(channels and block from the config), otherwise musical descriptors.

  --benchmark N         time N train windows and project the full cost; writes nothing else
  --per-genre N         pilot on the first N train tracks of each genre; never publishes a pointer
  --compare-to CACHE    with --per-genre: compare every window to an existing cache (max error)
  --development         V2 development: all 7 857 selected FMA-small tracks, V1 test included
                        (docs/PROTOCOLE_V2.md); default name dev_<config stem>

Without --development, V1 test audio is never decoded. V2 test audio is never decoded.
Per-track results are resumable and keyed by full provenance.
"""

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

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ("scripts/extract_features.py", "src/music_genre/audio/processing.py",
           "src/music_genre/features/descriptors.py", "src/music_genre/features/hpss.py",
           "src/music_genre/features/logmel.py", "src/music_genre/features/vit.py",
           "src/music_genre/models/vit.py")
WORKER = None


def build_extractor(config):
    if "checkpoint_directory" in config:
        from music_genre.features.vit import MusicalViT
        return MusicalViT(config, ROOT)
    from music_genre.features.descriptors import MusicalDescriptors
    return MusicalDescriptors(config)


def initialize(config):
    global WORKER
    # Worker lifetime: keep the limiter so numerical libraries use one thread, as in V1.
    WORKER = (build_extractor(config), threadpool_limits(limits=1))


def extract_track(task):
    row, folder, key, splits = task
    if row["split"] not in splits:
        raise ValueError("Test extraction forbidden")
    extractor = WORKER[0]
    path = (ROOT / "data/raw" / row["audio_path"]).resolve()
    if not path.is_relative_to((ROOT / "data/raw").resolve()) or sha256_file(path) != row["source_sha256"]:
        raise ValueError(f"Audio source changed: {row['track_id']}")
    target = Path(folder) / "tracks" / f"{row['track_id']}.npz"
    if target.exists():
        with np.load(target, allow_pickle=False) as saved:
            if str(saved["cache_key"]) != key or str(saved["source_sha256"]) != row["source_sha256"]:
                raise ValueError("Stale track cache")
            x, starts, valid = saved["x"], saved["starts"], saved["valid"]
    else:
        config = extractor.config
        audio = load_audio(path, sample_rate=config["sample_rate"])
        windows = segment_audio(audio, tuple(config["offsets_seconds"]), config["window_seconds"])
        if not windows:
            raise ValueError("No audio windows")
        x = np.stack([extractor.extract(window) for window in windows])
        starts = np.array([w.start_seconds for w in windows], dtype=np.float64)
        valid = np.array([w.valid_samples / w.sample_rate for w in windows], dtype=np.float64)
        temporary = target.with_suffix(".partial.npz")
        np.savez(temporary, x=x, starts=starts, valid=valid, cache_key=key,
                 source_sha256=row["source_sha256"])
        temporary.replace(target)
    if (x.shape != (len(starts), len(extractor.names)) or not len(starts) or len(starts) != len(valid)
            or not np.isfinite(x).all() or not np.all(valid > 0) or len(set(starts)) != len(starts)):
        raise ValueError("Invalid cached windows")
    return row, x, starts, valid


def selected_rows(labels, splits=("train", "validation")):
    pointer = json.loads((ROOT / "data/manifests/latest_baseline.json").read_text())
    manifest = (ROOT / pointer["folder"] / "manifest.csv").resolve()
    if (not manifest.is_relative_to((ROOT / "data/manifests").resolve())
            or sha256_file(manifest) != pointer["manifest_sha256"]):
        raise ValueError("Baseline manifest changed")
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] in splits]
    if any(r["status"] != "valid" or r["decision"] != "keep" for r in rows):
        raise ValueError("Unexpected non-selected track")
    if any(int(r["label_index"]) != labels.index(r["genre"]) for r in rows):
        raise ValueError("Label mapping mismatch")
    return pointer, rows


def pilot(rows, labels, per_genre):
    chosen = []
    for label in labels:
        chosen += sorted((r for r in rows if r["split"] == "train" and r["genre"] == label),
                         key=lambda r: r["track_id"])[:per_genre]
    return chosen


def compare_to_cache(cache, results):
    folder = (ROOT / cache).resolve()
    if not folder.is_relative_to((ROOT / "data/features").resolve()):
        raise ValueError("Comparison cache outside data/features")
    with (folder / "index.csv").open(encoding="utf-8", newline="") as stream:
        index = {(r["track_id"], float(r["start_seconds"])): int(r["row_index"]) for r in csv.DictReader(stream)}
    matrix = np.load(folder / "features.npy", mmap_mode="r", allow_pickle=False)
    worst = 0.0
    for row, x, starts, _ in results:
        reference = matrix[[index[(row["track_id"], float(s))] for s in starts]]
        worst = max(worst, float(np.max(np.abs(reference - x))))
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", required=True, help="Project-relative representation config")
    parser.add_argument("--name", help="Cache prefix and pointer name (default: config file stem)")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--benchmark", type=int, metavar="N")
    parser.add_argument("--per-genre", type=int, metavar="N")
    parser.add_argument("--compare-to", metavar="CACHE")
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be in 1..8")
    if args.compare_to and not args.per_genre:
        parser.error("--compare-to requires --per-genre")
    config_path = (ROOT / args.config).resolve()
    if not config_path.is_relative_to((ROOT / "configs").resolve()):
        raise ValueError("Configuration outside configs")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    name = args.name or ("dev_" if args.development else "") + config_path.stem
    labels = json.loads((ROOT / "configs/labels.json").read_text())["labels"]
    splits = ("train", "validation", "test") if args.development else ("train", "validation")
    pointer, rows = selected_rows(labels, splits)
    libraries = ("numpy", "scipy", "librosa", "soundfile", "torch", "timm", "safetensors")
    provenance = {"config": config, "manifest_sha256": pointer["manifest_sha256"], "labels": labels,
                  "splits": list(splits), "python": platform.python_version(),
                  "versions": {lib: version(lib) for lib in libraries},
                  "source_sha256": {path: sha256_file(ROOT / path) for path in SOURCES}}
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    if args.benchmark:
        with threadpool_limits(limits=1):
            extractor = build_extractor(config)
            sample = pilot(rows, labels, max(1, -(-args.benchmark // len(labels))))[:args.benchmark]
            seconds = []
            for row in [sample[0], *sample]:  # first call is warm-up
                audio = load_audio(ROOT / "data/raw" / row["audio_path"], sample_rate=config["sample_rate"])
                window = segment_audio(audio, (10.0,), config["window_seconds"])[0]
                started = time.perf_counter()
                extractor.extract(window)
                seconds.append(time.perf_counter() - started)
        median = float(np.median(seconds[1:]))
        windows = len(rows) * len(config["offsets_seconds"])
        report = {"name": name, "cache_key": key, "windows_timed": len(seconds) - 1,
                  "median_features_seconds_per_window": median, "train_validation_windows": windows,
                  "projected_hours_one_worker": median * windows / 3600,
                  "created_at_utc": datetime.now(UTC).isoformat(), "test_audio_loaded": False}
        (ROOT / "reports/benchmarks").mkdir(exist_ok=True)
        write_json(ROOT / f"reports/benchmarks/{name}.json", report)
        print(json.dumps(report, indent=2))
        return
    if args.per_genre:
        rows = pilot(rows, labels, args.per_genre)
    cache = ROOT / "data/features" / ((f"{name}-pilot-" if args.per_genre else f"{name}-") + key[:16])
    (cache / "tracks").mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    print(f"Cache: {cache}; tracks: {len(rows)}; splits: {', '.join(splits)}", flush=True)
    results = []
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(config,)) as pool:
            for count, result in enumerate(pool.map(extract_track, ((r, str(cache), key, splits) for r in rows),
                                                    chunksize=4), start=1):
                results.append(result)
                if count % 250 == 0:
                    print(f"{name}: {count}/{len(rows)} tracks", flush=True)
    except (ValueError, RuntimeError, OSError) as error:
        write_json(cache / "failure.json", {"error": str(error), "cache_key": key})
        raise
    if args.per_genre:
        summary = {"pilot_tracks": len(results), "cache": cache.relative_to(ROOT).as_posix()}
        if args.compare_to:
            summary.update(compare_to=args.compare_to, max_abs_error=compare_to_cache(args.compare_to, results))
        print(json.dumps(summary, indent=2))
        return
    extractor = build_extractor(config)
    index = [{"row_index": position, "track_id": row["track_id"], "window_id": window_id,
              "start_seconds": float(start), "valid_seconds": float(duration),
              "padded_seconds": max(0, config["window_seconds"] - float(duration)),
              "split": row["split"], "label_index": row["label_index"],
              "artist_id": row["artist_id"], "source_sha256": row["source_sha256"]}
             for position, (row, window_id, start, duration) in enumerate(
                 (row, w, s, d) for row, _, starts, valid in results
                 for w, (s, d) in enumerate(zip(starts, valid, strict=True)))]
    matrix = np.concatenate([x for _, x, _, _ in results])
    np.save(cache / "features.npy", matrix, allow_pickle=False)
    with (cache / "index.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(index[0]))
        writer.writeheader()
        writer.writerows(index)
    report = {"schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(), "cache_key": key,
              "provenance": provenance, "tracks": len(rows), "matrix_shape": list(matrix.shape),
              "feature_names": extractor.names, "feature_families": extractor.families,
              "elapsed_seconds": time.perf_counter() - started,
              "features_sha256": sha256_file(cache / "features.npy"),
              "index_sha256": sha256_file(cache / "index.csv"),
              "test_audio_loaded": args.development, "v2_test_audio_loaded": False}
    write_json(cache / "report.json", report)
    write_json(ROOT / f"data/features/latest_{name}.json",
               {"folder": cache.relative_to(ROOT).as_posix(), "report_sha256": sha256_file(cache / "report.json")})
    print(json.dumps({k: report[k] for k in ("tracks", "matrix_shape", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
