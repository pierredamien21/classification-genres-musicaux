"""Build full fusion and prespecified ablations from verified existing caches.

--development fuses the V2 development caches (dev_vit, dev_descriptors: V1 train, validation
and test) into dev_<variant>.
"""

import argparse
import csv
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from music_genre.data.audit import sha256_file, write_json
from music_genre.features.fusion import concatenate_features

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {"fusion": None, "fusion_no_rhythm": "rhythm", "fusion_no_harmony": "harmony"}


def load_cache(name, splits=("train", "validation")):
    pointer = json.loads((ROOT / f"data/features/latest_{name}.json").read_text(encoding="utf-8"))
    cache = (ROOT / pointer["folder"]).resolve()
    if not cache.is_relative_to((ROOT / "data/features").resolve()):
        raise ValueError("Cache outside feature directory")
    if sha256_file(cache / "report.json") != pointer["report_sha256"]:
        raise ValueError("Parent feature report changed")
    report = json.loads((cache / "report.json").read_text(encoding="utf-8"))
    for filename, key in (("features.npy", "features_sha256"), ("index.csv", "index_sha256")):
        if sha256_file(cache / filename) != report[key]:
            raise ValueError(f"Parent feature artifact changed: {filename}")
    if report["provenance"]["splits"] != list(splits) or report["test_audio_loaded"] != ("test" in splits):
        raise ValueError("Parent cache covers other splits")
    with (cache / "index.csv").open(encoding="utf-8", newline="") as stream:
        index = list(csv.DictReader(stream))
    return pointer, cache, report, index, np.load(cache / "features.npy", allow_pickle=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VARIANTS, default="fusion")
    parser.add_argument("--development", action="store_true")
    args = parser.parse_args()
    prefix = "dev_" if args.development else ""
    splits = ("train", "validation", "test") if args.development else ("train", "validation")
    output = prefix + args.variant
    started = time.perf_counter()
    dp, dc, dr, di, dx = load_cache(prefix + "descriptors", splits)
    vp, _, vr, vi, vx = load_cache(prefix + "vit", splits)
    for key in ("manifest_sha256", "labels", "splits"):
        if dr["provenance"][key] != vr["provenance"][key]:
            raise ValueError(f"Incompatible parent caches: {key}")
    if dr["index_sha256"] != vr["index_sha256"]:
        raise ValueError("Parent window indexes differ")
    if args.development:
        # V2 puts the V1 test split in development; fusion.py is pinned by the V1 lock, so its
        # train/validation guard is passed relabelled copies (load_cache already checked the splits).
        vi, di = ([{**row, "split": "train"} for row in rows] for rows in (vi, di))
    matrix, names, families, selected = concatenate_features(
        vx, dx, vi, di, vr["feature_names"], dr["feature_names"],
        dr["feature_families"], VARIANTS[args.variant],
    )
    config = {
        "schema_version": 1, "name": args.variant, "operation": "concatenate_columns",
        "order": ["vit", "descriptors"], "excluded_family": VARIANTS[args.variant],
        "descriptor_columns": selected, "standardization": "none_in_cache_train_only_in_classifier",
        "vit": vr["provenance"]["config"], "descriptors": dr["provenance"]["config"],
    }
    provenance = {
        "config": config, "parents": {"vit": vp, "descriptors": dp},
        "manifest_sha256": vr["provenance"]["manifest_sha256"],
        "labels": vr["provenance"]["labels"], "splits": list(splits),
        "numpy_version": np.__version__,
        "source_sha256": {name: sha256_file(ROOT / name) for name in (
            "scripts/build_fusion_features.py", "src/music_genre/features/fusion.py",
        )},
    }
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    cache = ROOT / "data/features" / f"{output}-{key[:16]}"
    if (cache / "report.json").exists():
        existing = json.loads((cache / "report.json").read_text(encoding="utf-8"))
        if (existing["provenance"] != provenance or existing["cache_key"] != key
                or sha256_file(cache / "features.npy") != existing["features_sha256"]
                or sha256_file(cache / "index.csv") != existing["index_sha256"]
                or existing["index_sha256"] != dr["index_sha256"]
                or existing["feature_names"] != names or existing["feature_families"] != families
                or not np.array_equal(np.load(cache / "features.npy", allow_pickle=False), matrix)):
            raise ValueError("Existing fusion cache changed; refusing to overwrite")
        write_json(ROOT / f"data/features/latest_{output}.json", {
            "folder": cache.relative_to(ROOT).as_posix(),
            "report_sha256": sha256_file(cache / "report.json"),
        })
        print(json.dumps({"variant": args.variant, "reused": True, "shape": list(matrix.shape)}))
        return
    cache.mkdir(parents=True, exist_ok=True)
    matrix_path = cache / "features.partial.npy"
    np.save(matrix_path, matrix, allow_pickle=False)
    matrix_path.replace(cache / "features.npy")
    (cache / "index.csv").write_bytes((dc / "index.csv").read_bytes())
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(),
        "cache_key": key, "provenance": provenance,
        "tracks": len({row["track_id"] for row in vi}), "matrix_shape": list(matrix.shape),
        "feature_names": names, "feature_families": families,
        "elapsed_seconds": time.perf_counter() - started,
        "features_sha256": sha256_file(cache / "features.npy"),
        "index_sha256": sha256_file(cache / "index.csv"),
        "audio_loaded": False, "test_audio_loaded": args.development,
    }
    write_json(cache / "report.json", report)
    write_json(ROOT / f"data/features/latest_{output}.json", {
        "folder": cache.relative_to(ROOT).as_posix(),
        "report_sha256": sha256_file(cache / "report.json"),
    })
    print(json.dumps({"variant": args.variant, "shape": list(matrix.shape),
                      "seconds": report["elapsed_seconds"], "cache": str(cache)}, indent=2))


if __name__ == "__main__":
    main()
