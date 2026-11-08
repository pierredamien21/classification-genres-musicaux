"""Training-only feasibility and equivalence checks for the block-six HPSS candidate."""

import csv
import json
import platform
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from diagnose_review_audio import load_cache, read_json
from threadpoolctl import threadpool_limits

from music_genre.audio.processing import load_audio, segment_audio
from music_genre.data.audit import sha256_file, write_json
from music_genre.features.hpss import MusicalHpssViT
from music_genre.features.hpss_block6 import MusicalHpssBlock6
from music_genre.features.logmel import log_mel

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    config = read_json(ROOT / "configs/vit_block6_hpss.json")
    experiment = read_json(ROOT / "configs/block6_hpss_experiment.json")
    bench = experiment["benchmark"]
    if (bench["split"] != "train" or bench["tracks_per_genre"] != 8
            or bench["offset_seconds"] != 10. or bench["warmup_windows"] != 1
            or bench["workers"] != 1 or config["threads_per_worker"] != 1):
        raise ValueError("Unsupported benchmark protocol")
    for field in ("original_logmel_absolute_tolerance_db", "reference_embedding_absolute_tolerance",
                  "serial_extraction_budget_seconds", "estimate_safety_factor"):
        if not np.isfinite(bench[field]) or bench[field] <= 0:
            raise ValueError("Invalid benchmark limit")
    baseline = read_json(ROOT / "data/manifests/latest_baseline.json")
    manifest = (ROOT / baseline["folder"] / "manifest.csv").resolve()
    if not manifest.is_relative_to((ROOT / "data/manifests").resolve()) or sha256_file(manifest) != baseline["manifest_sha256"]:
        raise ValueError("Baseline manifest changed")
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = [r for r in csv.DictReader(stream) if r["split"] in ("train", "validation")]
    labels = read_json(ROOT / "configs/labels.json")["labels"]
    if (len({r["track_id"] for r in rows}) != len(rows)
            or any(r["status"] != "valid" or r["decision"] != "keep"
                   or labels[int(r["label_index"])] != r["genre"] for r in rows)):
        raise ValueError("Invalid selected cohort")
    counts = {split: sum(r["split"] == split for r in rows) for split in ("train", "validation")}
    if counts != {"train": 6299, "validation": 780}:
        raise ValueError("Unexpected cohort counts")
    caches = {name: load_cache(name, baseline, labels) for name in ("vit_block6", "vit_hpss")}
    if caches["vit_block6"][1]["index_sha256"] != caches["vit_hpss"][1]["index_sha256"]:
        raise ValueError("Reference windows differ")
    varying = {"vit_block6": {"name", "channels", "hpss"},
               "vit_hpss": {"name", "feature_block", "pooling", "terminal_norm"}}
    for name, ignored in varying.items():
        parent_config = caches[name][1]["provenance"]["config"]
        if {k: v for k, v in config.items() if k not in ignored} != {k: v for k, v in parent_config.items() if k not in ignored}:
            raise ValueError(f"Uncontrolled configuration difference: {name}")
        if caches[name][1]["test_audio_loaded"] or caches[name][1]["provenance"]["splits"] != ["train", "validation"]:
            raise ValueError("Reference includes test")
    classifier = read_json(ROOT / "configs/baseline_vit_block6_hpss.json")
    for name in ("vit_block6", "vit_hpss"):
        other = read_json(ROOT / f"configs/baseline_{name}.json")
        if {k: v for k, v in classifier.items() if k != "name"} != {k: v for k, v in other.items() if k != "name"}:
            raise ValueError("Classifier protocol differs")
    rng = np.random.default_rng(bench["seed"])
    selected = []
    for label in labels:
        group = [r for r in rows if r["split"] == "train" and r["genre"] == label]
        selected.extend(group[int(i)] for i in rng.choice(len(group), bench["tracks_per_genre"], replace=False))
    old_path = ROOT / "reports/VIT_HPSS_BENCHMARK.json"
    if sha256_file(old_path) != caches["vit_hpss"][1]["benchmark_sha256"]:
        raise ValueError("Historical HPSS benchmark changed")
    if [r["track_id"] for r in selected] != [r["track_id"] for r in read_json(old_path)["samples"]]:
        raise ValueError("Benchmark selection differs from HPSS reference")
    candidate = MusicalHpssBlock6(config, ROOT)
    full = MusicalHpssViT(caches["vit_hpss"][1]["provenance"]["config"], ROOT)
    samples = []
    with threadpool_limits(limits=1):
        for number, row in enumerate([selected[0], *selected]):
            key = row["track_id"]
            path = (ROOT / "data/raw" / row["audio_path"]).resolve()
            if not path.is_relative_to((ROOT / "data/raw/fma_small").resolve()) or sha256_file(path) != row["source_sha256"]:
                raise ValueError("Benchmark source changed or outside dataset")
            t0 = perf_counter()
            audio = load_audio(path, config["sample_rate"])
            window = segment_audio(audio, (bench["offset_seconds"],), config["window_seconds"])[0]
            t1 = perf_counter()
            channels = candidate.spectrograms(window)
            t2 = perf_counter()
            x = candidate.encoder.extract(channels, config["db_min"], config["db_max"])
            t3 = perf_counter()
            full_x = full.encoder.extract(channels, config["db_min"], config["db_max"])
            t4 = perf_counter()
            if x.shape != (192,) or not np.isfinite(x).all():
                raise ValueError("Invalid candidate embedding")
            repeated_x = candidate.encoder.extract(np.repeat(channels[:1], 3, axis=0), config["db_min"], config["db_max"])
            errors = {"original_logmel": float(np.max(np.abs(channels[0] - log_mel(window, config["logmel"]))))}
            for name, actual in (("vit_block6", repeated_x), ("vit_hpss", full_x)):
                ref = caches[name][2][key, 1]
                if (any(ref[k] != row[k] for k in ("split", "artist_id", "label_index", "source_sha256"))
                        or float(ref["start_seconds"]) != window.start_seconds
                        or float(ref["valid_seconds"]) != window.valid_samples / window.sample_rate):
                    raise ValueError("Benchmark window differs from reference")
                errors[name] = float(np.max(np.abs(actual - caches[name][3][int(ref["row_index"])])))
            if (errors["original_logmel"] > bench["original_logmel_absolute_tolerance_db"]
                    or max(errors["vit_block6"], errors["vit_hpss"]) > bench["reference_embedding_absolute_tolerance"]):
                raise ValueError("Reference equivalence failed")
            sample = {"track_id": key, "genre": row["genre"], "split": row["split"],
                      "source_sha256": row["source_sha256"], "offset_seconds": window.start_seconds,
                      "dimensions": len(x), "decode_seconds": t1-t0, "hpss_logmel_seconds": t2-t1,
                      "block6_seconds": t3-t2, "block12_seconds": t4-t3,
                      "candidate_seconds": t3-t0, "max_abs_errors": errors}
            if number:
                samples.append(sample)
                if number % 16 == 0:
                    print(f"Verified {number}/64 training windows", flush=True)
            else:
                warmup = sample
    times = {field: {"mean": float(np.mean([s[field] for s in samples])),
                     "median": float(np.median([s[field] for s in samples])),
                     "p95": float(np.percentile([s[field] for s in samples], 95))}
             for field in ("decode_seconds", "hpss_logmel_seconds", "block6_seconds", "block12_seconds", "candidate_seconds")}
    serial = len(rows) * (times["decode_seconds"]["mean"] + 3 * (times["hpss_logmel_seconds"]["mean"] + times["block6_seconds"]["mean"]))
    guarded = serial * bench["estimate_safety_factor"]
    source_paths = ["scripts/benchmark_vit_block6_hpss.py", "scripts/diagnose_review_audio.py",
                    "src/music_genre/features/hpss_block6.py", "src/music_genre/features/hpss.py",
                    "src/music_genre/models/vit_channels.py", "src/music_genre/models/vit.py",
                    "src/music_genre/audio/processing.py", "src/music_genre/features/logmel.py",
                    "src/music_genre/data/audit.py"]
    config_paths = ["configs/vit_block6_hpss.json", "configs/baseline_vit_block6_hpss.json", "configs/block6_hpss_experiment.json"]
    result = {"status": "complete", "config": config, "experiment": experiment,
              "baseline": baseline, "parents": {name: cache[0] for name, cache in caches.items()},
              "sample_count": len(samples), "samples": samples, "warmup": warmup, "timings": times,
              "max_abs_errors": {k: max(s["max_abs_errors"][k] for s in samples) for k in warmup["max_abs_errors"]},
              "projection": {"tracks": len(rows), "windows": len(rows)*3, "workers": 1,
                             "serial_seconds": serial, "guarded_seconds": guarded,
                             "within_budget": guarded <= bench["serial_extraction_budget_seconds"],
                             "excludes": "initialization, cache writes, checksum checks, external interruptions"},
              "test_audio_loaded": False, "test_evaluated": False, "full_extraction_completed": False,
              "classifier_trained": False, "elapsed_seconds_including_checks": perf_counter()-started,
              "python": platform.python_version(),
              "versions": {p: version(p) for p in ("numpy", "scipy", "librosa", "soundfile", "torch", "timm", "safetensors")},
              "source_sha256": {p: sha256_file(ROOT / p) for p in source_paths},
              "config_sha256": {p: sha256_file(ROOT / p) for p in config_paths}}
    target = ROOT / "reports/VIT_BLOCK6_HPSS_BENCHMARK.json"
    write_json(target, result)
    write_json(ROOT / "reports/latest_vit_block6_hpss_benchmark.json", {
        "report": target.relative_to(ROOT).as_posix(), "report_sha256": sha256_file(target)})
    print(json.dumps({k: result[k] for k in ("status", "sample_count", "timings", "max_abs_errors", "projection")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
