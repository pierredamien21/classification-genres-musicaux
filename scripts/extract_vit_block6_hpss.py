"""Resumable HPSS/block-six extraction; publish only a fully verified cache."""

import argparse
import csv
import hashlib
import json
import msvcrt
import os
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from diagnose_review_audio import load_cache, read_json
from threadpoolctl import threadpool_limits

from music_genre.audio.processing import load_audio, segment_audio
from music_genre.data.audit import sha256_file, write_json
from music_genre.features.hpss_block6 import MusicalHpssBlock6
from music_genre.features.track_cache import load_track_cache, save_track_cache

ROOT = Path(__file__).resolve().parents[1]
WORKER = None


def initialize(config):
    global WORKER
    WORKER = (MusicalHpssBlock6(config, ROOT), threadpool_limits(limits=1))


def extract_track(task):
    row, expected, folder, key = task
    if row["split"] not in ("train", "validation"):
        raise ValueError("Test extraction forbidden")
    path = (ROOT / "data/raw" / row["audio_path"]).resolve()
    if not path.is_relative_to((ROOT / "data/raw/fma_small").resolve()) or sha256_file(path) != row["source_sha256"]:
        raise ValueError("Audio source changed")
    target = Path(folder) / "tracks" / f"{row['track_id']}.npz"
    saved = load_track_cache(target, key, row["source_sha256"])
    resumed = saved is not None
    if resumed:
        x, valid = saved
    else:
        extractor = WORKER[0]
        audio = load_audio(path, extractor.config["sample_rate"])
        windows = segment_audio(audio, tuple(extractor.config["offsets_seconds"]), extractor.config["window_seconds"])
        if [w.start_seconds for w in windows] != [0., 10., 20.]:
            raise ValueError("Missing reference windows")
        x = np.stack([extractor.extract(w) for w in windows])
        valid = np.array([w.valid_samples / w.sample_rate for w in windows])
    if not np.array_equal(valid, [float(r["valid_seconds"]) for r in expected]):
        raise ValueError("Decoded durations differ from reference")
    if not resumed:
        save_track_cache(target, key, row["source_sha256"], x, valid)
    return row["track_id"], x, resumed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot", action="store_true", help="Only 64 benchmark train tracks; never publish a complete cache")
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        parser.error("workers must be in 1..4")
    bp = read_json(ROOT / "reports/latest_vit_block6_hpss_benchmark.json")
    benchmark_path = (ROOT / bp["report"]).resolve()
    if not benchmark_path.is_relative_to((ROOT / "reports").resolve()) or sha256_file(benchmark_path) != bp["report_sha256"]:
        raise ValueError("Benchmark report changed")
    benchmark = read_json(benchmark_path)
    if (benchmark["status"] != "complete" or benchmark["sample_count"] != 64
            or not benchmark["projection"]["within_budget"] or benchmark["test_audio_loaded"]):
        raise ValueError("Successful train-only benchmark required")
    for field in ("source_sha256", "config_sha256"):
        for path, expected in benchmark[field].items():
            if sha256_file(ROOT / path) != expected:
                raise ValueError(f"Benchmarked source/config changed: {path}")
    from importlib.metadata import version
    for package, expected in benchmark["versions"].items():
        if version(package) != expected:
            raise ValueError("Benchmarked library version changed")
    baseline = read_json(ROOT / "data/manifests/latest_baseline.json")
    if baseline != benchmark["baseline"]:
        raise ValueError("Benchmarked cohort changed")
    manifest = (ROOT / baseline["folder"] / "manifest.csv").resolve()
    if not manifest.is_relative_to((ROOT / "data/manifests").resolve()) or sha256_file(manifest) != baseline["manifest_sha256"]:
        raise ValueError("Manifest changed")
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] in ("train", "validation")]
    labels = read_json(ROOT / "configs/labels.json")["labels"]
    parents = {name: load_cache(name, baseline, labels) for name in ("vit_block6", "vit_hpss")}
    if any(parents[n][0] != benchmark["parents"][n] for n in parents):
        raise ValueError("Reference cache changed since benchmark")
    ref = parents["vit_hpss"]
    if ref[1]["index_sha256"] != parents["vit_block6"][1]["index_sha256"]:
        raise ValueError("Reference windows differ")
    index = list(ref[2].values())
    if (len(rows) != 7079 or len(index) != 21237 or len({r["track_id"] for r in rows}) != len(rows)
            or {r["track_id"] for r in rows} != {r["track_id"] for r in index}):
        raise ValueError("Incomplete cohort")
    grouped = {}
    for r in index:
        grouped.setdefault(r["track_id"], []).append(r)
    for row in rows:
        expected = grouped[row["track_id"]]
        if (row["status"] != "valid" or row["decision"] != "keep"
                or [int(r["window_id"]) for r in expected] != [0, 1, 2]
                or [float(r["start_seconds"]) for r in expected] != [0., 10., 20.]
                or any(r[k] != row[k] for r in expected for k in ("split", "artist_id", "label_index", "source_sha256"))):
            raise ValueError("Reference index differs from selected manifest")
    config = benchmark["config"]
    provenance = {"config": config, "manifest_sha256": baseline["manifest_sha256"],
                  "labels": labels, "splits": ["train", "validation"], "parents": benchmark["parents"],
                  "versions": benchmark["versions"], "benchmark": bp,
                  "source_sha256": {**benchmark["source_sha256"], **{p: sha256_file(ROOT / p) for p in (
                      "scripts/extract_vit_block6_hpss.py", "src/music_genre/features/track_cache.py")}}}
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    cache = ROOT / "data/features" / f"vit-block6-hpss-{key[:16]}"
    (cache / "tracks").mkdir(parents=True, exist_ok=True)
    # Windows byte lock releases automatically when the process exits.
    with (cache / "extraction.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        execute(args, rows, grouped, index, ref, cache, key, provenance, bp)


def execute(args, rows, grouped, index, reference, cache, key, provenance, benchmark_pointer):
    if (cache / "report.json").exists() and not args.pilot:
        complete = read_json(cache / "report.json")
        if (complete["cache_key"] != key or complete["provenance"] != provenance
                or complete["index_sha256"] != reference[1]["index_sha256"]
                or any(sha256_file(cache / filename) != complete[field] for filename, field in (
                    ("features.npy", "features_sha256"), ("index.csv", "index_sha256")))):
            raise ValueError("Completed cache changed")
        write_json(ROOT / "data/features/latest_vit_block6_hpss.json", {
            "folder": cache.relative_to(ROOT).as_posix(), "report_sha256": sha256_file(cache / "report.json")})
        print("Verified completed cache; preserving extraction report", flush=True)
        return
    write_json(cache / "provenance.json", provenance)
    if args.pilot:
        ids = {r["track_id"] for r in read_json(ROOT / benchmark_pointer["report"])["samples"]}
        rows = [r for r in rows if r["track_id"] in ids and r["split"] == "train"]
        if len(rows) != 64:
            raise ValueError("Incomplete pilot selection")
    progress = {"status": "running", "pilot": args.pilot, "cache": cache.relative_to(ROOT).as_posix(),
                "cache_key": key, "total_tracks": len(rows), "processed_tracks": 0, "resumed_tracks": 0,
                "workers": args.workers, "started_at_utc": datetime.now(UTC).isoformat(),
                "test_audio_loaded": False}
    progress_path = ROOT / "reports/VIT_BLOCK6_HPSS_PROGRESS.json"
    write_json(progress_path, progress)
    started = perf_counter()
    matrices = {}
    os.environ.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
    print(f"Cache: {cache}; {len(rows)} tracks; workers={args.workers}", flush=True)
    try:
        with ProcessPoolExecutor(max_workers=args.workers, initializer=initialize, initargs=(provenance["config"],)) as pool:
            tasks = ((r, grouped[r["track_id"]], str(cache), key) for r in rows)
            for count, (track, x, resumed) in enumerate(pool.map(extract_track, tasks, chunksize=1), 1):
                matrices[track] = x
                progress.update(processed_tracks=count, resumed_tracks=progress["resumed_tracks"]+int(resumed),
                                elapsed_seconds=perf_counter()-started, updated_at_utc=datetime.now(UTC).isoformat())
                if count % 25 == 0 or count == len(rows):
                    write_json(progress_path, progress)
                    print(f"Verified {count}/{len(rows)}; resumed={progress['resumed_tracks']}; {progress['elapsed_seconds']:.1f}s", flush=True)
        if not args.pilot:
            for path, expected in provenance["source_sha256"].items():
                if sha256_file(ROOT / path) != expected:
                    raise ValueError("Extraction source changed while running")
            matrix = np.stack([matrices[r["track_id"]][int(r["window_id"])] for r in index])
            np.save(cache / "features.npy", matrix, allow_pickle=False)
            (cache / "index.csv").write_bytes((ROOT / reference[0]["folder"] / "index.csv").read_bytes())
            report = {"schema_version": 1, "cache_key": key, "provenance": provenance,
                      "created_at_utc": datetime.now(UTC).isoformat(), "tracks": len(rows),
                      "matrix_shape": list(matrix.shape), "feature_names": [f"vit_patch_mean_{i:03d}" for i in range(192)],
                      "feature_families": ["vit"]*192, "workers": args.workers,
                      "elapsed_seconds": perf_counter()-started, "resumed_tracks": progress["resumed_tracks"],
                      "benchmark_sha256": benchmark_pointer["report_sha256"], "test_audio_loaded": False,
                      "index_sha256": sha256_file(cache / "index.csv"), "features_sha256": sha256_file(cache / "features.npy")}
            if report["index_sha256"] != reference[1]["index_sha256"]:
                raise ValueError("Completed index differs from reference")
            write_json(cache / "report.json", report)
            write_json(ROOT / "data/features/latest_vit_block6_hpss.json", {
                "folder": cache.relative_to(ROOT).as_posix(), "report_sha256": sha256_file(cache / "report.json")})
        progress["status"] = "pilot_complete" if args.pilot else "complete"
        write_json(progress_path, progress)
    except BaseException as error:
        progress.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
        write_json(progress_path, progress)
        raise


if __name__ == "__main__":
    main()
