"""Derive central-window embeddings without decoding audio or rerunning the ViT."""

import csv
import hashlib
import json
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from build_fusion_features import load_cache

from music_genre.data.audit import sha256_file, write_json
from music_genre.features.windows import central_window_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    began = time.perf_counter()
    pointer, _, parent, index, matrix = load_cache("vit")
    positions, selected_index = central_window_rows(index)
    selected = matrix[positions]
    if selected.shape != (len(selected_index), 192) or not np.isfinite(selected).all():
        raise ValueError("Invalid central embeddings")
    config = {**parent["provenance"]["config"], "name": "vit_frozen_central_v1", "offsets_seconds": [10.0]}
    provenance = {
        "config": config, "parent": pointer, "selection": "one_start10_window_per_track",
        "manifest_sha256": parent["provenance"]["manifest_sha256"],
        "labels": parent["provenance"]["labels"], "splits": ["train", "validation"],
        "numpy_version": np.__version__,
        "source_sha256": {name: sha256_file(ROOT / name) for name in (
            "scripts/build_vit_central.py", "scripts/build_fusion_features.py",
            "src/music_genre/features/windows.py",
        )},
    }
    key = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode()).hexdigest()
    cache = ROOT / "data/features" / f"vit-central-{key[:16]}"
    if (cache / "report.json").exists():
        report = json.loads((cache / "report.json").read_text(encoding="utf-8"))
        if report["provenance"] != provenance:
            raise ValueError("Central cache provenance changed")
        for filename, field in (("features.npy", "features_sha256"), ("index.csv", "index_sha256")):
            if sha256_file(cache / filename) != report[field]:
                raise ValueError("Central cache artifact changed")
        if not np.array_equal(np.load(cache / "features.npy", allow_pickle=False), selected):
            raise ValueError("Central vectors differ from parent")
        with (cache / "index.csv").open(encoding="utf-8", newline="") as stream:
            if list(csv.DictReader(stream)) != selected_index:
                raise ValueError("Central index differs from parent subset")
    else:
        cache.mkdir(parents=True, exist_ok=True)
        temporary = cache / "features.partial.npy"
        np.save(temporary, selected, allow_pickle=False)
        temporary.replace(cache / "features.npy")
        with (cache / "index.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(selected_index[0]))
            writer.writeheader()
            writer.writerows(selected_index)
        report = {
            "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(),
            "cache_key": key, "provenance": provenance, "tracks": len(selected_index),
            "matrix_shape": list(selected.shape), "feature_names": parent["feature_names"],
            "feature_families": parent["feature_families"], "elapsed_seconds": time.perf_counter() - began,
            "features_sha256": sha256_file(cache / "features.npy"),
            "index_sha256": sha256_file(cache / "index.csv"), "audio_loaded": False, "test_audio_loaded": False,
        }
        write_json(cache / "report.json", report)
    write_json(ROOT / "data/features/latest_vit_central.json", {
        "folder": cache.relative_to(ROOT).as_posix(), "report_sha256": sha256_file(cache / "report.json"),
    })
    print(json.dumps({"shape": list(selected.shape), "cache": str(cache), "test_loaded": False}, indent=2))


if __name__ == "__main__":
    main()
