"""V2 phase 4: robustness of a nested-CV run to the degradations of configs/degradations_v2.json.

  python scripts/robustness_v2.py --run runs/v2-cv/dev_maest-layer12-<timestamp> --threads 8

Each outer fold is refitted with the C its run selected (plus the augmented windows when the run
used --augment). The clean predictions must reproduce the run's out-of-fold file; then every
degraded cache dev_<features>_<condition> is predicted with the same heads. Writes
runs/v2-robustness/<system>-<timestamp>/ with report.json and one out-of-fold file per condition.
Paired artist bootstrap (2 000 draws) for each condition minus clean. Development audio only.
"""

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_v2 import load_cache, make_fit

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.bootstrap import paired_artist_bootstrap
from music_genre.evaluation.crossval import refit_outer, track_mean

ROOT = Path(__file__).resolve().parents[1]


def columns(report, family):
    return slice(None) if not family else [i for i, f in enumerate(report["feature_families"]) if f == family]


def load_aligned(name, family, index, expected_folder=None):
    pointer, report, rows, x = load_cache(name)
    if expected_folder and pointer["folder"] != expected_folder:
        raise ValueError(f"{name} does not point to the cache the run used")
    if [(r["track_id"], r["start_seconds"]) for r in rows] != [(r["track_id"], r["start_seconds"]) for r in index]:
        raise ValueError(f"{name} is not aligned with the clean cache")
    return pointer, x[:, columns(report, family)]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run = (ROOT / args.run).resolve()
    if not run.is_relative_to((ROOT / "runs/v2-cv").resolve()):
        raise ValueError("Run outside runs/v2-cv")
    result = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if sha256_file(run / "oof_tracks.csv") != result["oof_tracks_sha256"]:
        raise ValueError("Out-of-fold file changed")
    labels = json.loads((ROOT / "configs/labels.json").read_text())["labels"]
    degradations = json.loads((ROOT / "configs/degradations_v2.json").read_text(encoding="utf-8"))
    features, family = result["features"], result["family"]
    pointer, report, index, x = load_cache(features)
    if pointer["folder"] != result["feature_cache"]:
        raise ValueError("Clean cache differs from the run's")
    x = x[:, columns(report, family)]
    augmented = bool(result.get("augment_cache"))
    if augmented:
        _, x_augment = load_aligned(f"{features}_augmentation", family, index, result["augment_cache"])
    with (ROOT / result["config"]["folds"]).open(encoding="utf-8", newline="") as stream:
        folds_by_track = {r["track_id"]: r for r in csv.DictReader(stream)}
    tracks = np.array([r["track_id"] for r in index])
    y = np.array([int(r["label_index"]) for r in index])
    folds = np.array([int(folds_by_track[t]["fold"]) for t in tracks])
    evaluate, pointers = {"clean": x}, {}
    for condition in degradations["conditions"]:
        pointers[condition], evaluate[condition] = load_aligned(f"{features}_{condition}", family, index)
    fit_x = np.hstack([x, x_augment]) if augmented else x
    if augmented:  # the augmented head reads only the first (clean-width) half at prediction time
        evaluate = {name: np.hstack([m, np.zeros_like(m)]) for name, m in evaluate.items()}
    selected = {r["fold"]: r["selected_c"] for r in result["outer_folds"]}
    with threadpool_limits(limits=args.threads):
        out = refit_outer(fit_x, y, folds, selected, make_fit(result["config"], labels, augmented), evaluate)
    with (run / "oof_tracks.csv").open(encoding="utf-8", newline="") as stream:
        saved = {r["track_id"]: np.array([float(r[f"p_{label}"]) for label in labels]) for r in csv.DictReader(stream)}
    per_track = {name: track_mean(tracks, y, probs) for name, probs in out.items()}
    ids, truth, clean = per_track["clean"]
    reproduction = float(max(np.max(np.abs(saved[t] - p)) for t, p in zip(ids, clean, strict=True)))
    if reproduction > 1e-6:
        raise ValueError(f"Refitted heads do not reproduce the run: {reproduction}")
    folder = ROOT / "runs/v2-robustness" / f"{result['system']}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    folder.mkdir(parents=True)
    conditions, predictions = {}, {}
    for name, (names, _, probs) in per_track.items():
        if not np.array_equal(names, ids):
            raise ValueError("Track order differs between conditions")
        predictions[name] = [{"track_id": t, "scores": p.tolist()} for t, p in zip(ids, probs, strict=True)]
        with (folder / f"oof_{name}.csv").open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream, lineterminator="\n")
            writer.writerow(["track_id", "fold", "label_index", *(f"p_{label}" for label in labels)])
            writer.writerows([t, folds_by_track[t]["fold"], int(k), *(repr(float(v)) for v in p)]
                             for t, k, p in zip(ids, truth, probs, strict=True))
        predicted = probs.argmax(axis=1)
        conditions[name] = {
            "macro_f1": float(f1_score(truth, predicted, average="macro")),
            "accuracy": float(accuracy_score(truth, predicted)),
            "changed_predictions": float(np.mean(predicted != clean.argmax(axis=1))),
            "mean_top_probability": float(probs.max(axis=1).mean()),
            "per_class_f1": dict(zip(labels, f1_score(truth, predicted, average=None).tolist(), strict=True)),
            "oof_sha256": sha256_file(folder / f"oof_{name}.csv"),
        }
    track_rows = [{"track_id": t, "artist_id": folds_by_track[t]["artist_id"], "label_index": int(k)}
                  for t, k in zip(ids, truth, strict=True)]
    boot = paired_artist_bootstrap(track_rows, predictions, labels, replicates=2000, seed=20261110,
                                   comparisons=[(name, "clean") for name in degradations["conditions"]])
    summary = {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(), "system": result["system"],
        "run": run.relative_to(ROOT).as_posix(), "run_report_sha256": sha256_file(run / "report.json"),
        "augmented_head": augmented, "degradations_sha256": sha256_file(ROOT / "configs/degradations_v2.json"),
        "degraded_caches": {k: v["folder"] for k, v in pointers.items()},
        "source_sha256": {p: sha256_file(ROOT / p) for p in (
            "scripts/robustness_v2.py", "scripts/cv_v2.py", "src/music_genre/evaluation/crossval.py")},
        "clean_reproduction_max_abs_error": reproduction, "tracks": len(ids), "conditions": conditions,
        "bootstrap": {"replicates": 2000, "seed": 20261110, "unit": "artist",
                      "artists": len(boot["artist_ids"]), "estimates": boot["estimates"],
                      "differences": boot["differences"]},
        "v2_test_audio_loaded": False,
    }
    write_json(folder / "report.json", summary)
    print(json.dumps({k: {m: round(v[m], 4) for m in ("macro_f1", "accuracy", "changed_predictions")}
                      for k, v in conditions.items()}, indent=1))


if __name__ == "__main__":
    main()
