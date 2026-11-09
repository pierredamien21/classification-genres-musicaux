"""Nested 5-fold artist-grouped cross-validation of a cached representation (V2 phase 3).

  python scripts/cv_v2.py --features dev_vit_block6_hpss
  python scripts/cv_v2.py --features dev_maest --family layer12 --config configs/cv_v2_mlp.json --threads 8

Reads data/features/latest_<features>.json (a --development cache: V1 train, validation and
test), the pre-registered folds (reports/v2/dev_folds.csv) and a CV config (logistic head by
default; "head": "mlp" switches to a one-hidden-layer MLP whose grid is its L2 alpha). Writes
runs/v2-cv/<features>-<timestamp>/ with report.json and the out-of-fold track probabilities.
No V2 test or out-of-domain audio is involved.
"""

import argparse
import csv
import json
import time
import warnings
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.crossval import macro_f1, nested_cv, track_mean

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/cv_v2.json"


def load_cache(name):
    pointer = json.loads((ROOT / f"data/features/latest_{name}.json").read_text(encoding="utf-8"))
    cache = (ROOT / pointer["folder"]).resolve()
    if not cache.is_relative_to((ROOT / "data/features").resolve()):
        raise ValueError("Cache outside feature directory")
    if sha256_file(cache / "report.json") != pointer["report_sha256"]:
        raise ValueError("Feature report changed")
    report = json.loads((cache / "report.json").read_text(encoding="utf-8"))
    for filename, key in (("features.npy", "features_sha256"), ("index.csv", "index_sha256")):
        if sha256_file(cache / filename) != report[key]:
            raise ValueError(f"Feature artifact changed: {filename}")
    if report["provenance"]["splits"] != ["train", "validation", "test"]:
        raise ValueError("Not a V2 development cache")
    with (cache / "index.csv").open(encoding="utf-8", newline="") as stream:
        index = list(csv.DictReader(stream))
    return pointer, report, index, np.load(cache / "features.npy", allow_pickle=False).astype(np.float64)


def classifier(config, value):
    if config.get("head", "logistic") == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(config["hidden_units"],), alpha=value, batch_size=config["batch_size"],
            learning_rate_init=config["learning_rate"], max_iter=config["max_iter"], tol=config["tolerance"],
            n_iter_no_change=config["n_iter_no_change"], early_stopping=False, random_state=config["seed"])
    return LogisticRegression(C=value, solver=config["solver"], max_iter=config["max_iter"],
                              tol=config["tolerance"], random_state=config["seed"],
                              class_weight=config["class_weight"])


def make_fit(config, labels):
    def fit_predict(x_fit, y_fit, x_eval, c):
        pipeline = Pipeline([("scale", StandardScaler()), ("classifier", classifier(config, c))])
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            pipeline.fit(x_fit, y_fit)
        if any(issubclass(w.category, ConvergenceWarning) for w in captured):
            raise RuntimeError(f"{c} did not converge")
        if not np.array_equal(pipeline.named_steps["classifier"].classes_, np.arange(len(labels))):
            raise ValueError("Unexpected class order")
        return pipeline.predict_proba(x_eval)
    return fit_predict


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", required=True)
    parser.add_argument("--family", help="Keep only the columns of this feature family (e.g. layer6)")
    parser.add_argument("--config", default=CONFIG)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    config = json.loads((ROOT / args.config).read_text(encoding="utf-8"))
    head = config.get("head", "logistic")
    labels = json.loads((ROOT / "configs/labels.json").read_text())["labels"]
    pointer, report, index, x = load_cache(args.features)
    system = args.features
    if args.family:
        columns = [i for i, f in enumerate(report["feature_families"]) if f == args.family]
        if not columns:
            raise ValueError(f"No columns in family {args.family}")
        x, system = x[:, columns], f"{args.features}-{args.family}"
    system += "" if head == "logistic" else f"-{head}"
    if report["provenance"]["labels"] != labels:
        raise ValueError("Class order changed")
    with (ROOT / config["folds"]).open(encoding="utf-8", newline="") as stream:
        folds_by_track = {r["track_id"]: r for r in csv.DictReader(stream)}
    if {r["track_id"] for r in index} != set(folds_by_track) or len(x) != len(index) or not np.isfinite(x).all():
        raise ValueError("Cache does not cover exactly the development tracks")
    for position, row in enumerate(index):
        fold = folds_by_track[row["track_id"]]
        if (int(row["row_index"]) != position or row["artist_id"] != fold["artist_id"]
                or labels[int(row["label_index"])] != fold["genre"]):
            raise ValueError(f"Cache row disagrees with the folds: {row['track_id']}")
    tracks = np.array([r["track_id"] for r in index])
    y = np.array([int(r["label_index"]) for r in index])
    folds = np.array([int(folds_by_track[t]["fold"]) for t in tracks])
    fit_predict = make_fit(config, labels)
    started = time.perf_counter()
    with threadpool_limits(limits=args.threads):
        oof, records = nested_cv(x, y, folds, tracks, config["c_grid"], fit_predict)
        plain = {}
        for c in config["c_grid"]:
            plain[c] = [macro_f1(tracks[folds == k], y[folds == k],
                                 fit_predict(x[folds != k], y[folds != k], x[folds == k], c))
                        for k in range(5)]
    final_c = max(sorted(config["c_grid"]), key=lambda c: np.mean(plain[c]))
    track_ids, truth, probabilities = track_mean(tracks, y, oof)
    fold_scores = [r["macro_f1"] for r in records]
    folder = ROOT / "runs/v2-cv" / f"{system}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    folder.mkdir(parents=True)
    with (folder / "oof_tracks.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["track_id", "fold", "label_index", *(f"p_{label}" for label in labels)])
        writer.writerows([t, folds_by_track[t]["fold"], int(k), *(repr(float(v)) for v in p)]
                         for t, k, p in zip(track_ids, truth, probabilities, strict=True))
    result = {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(), "system": system,
        "features": args.features, "family": args.family, "head": head,
        "config": config, "config_sha256": sha256_file(ROOT / args.config),
        "source_sha256": {p: sha256_file(ROOT / p) for p in (
            "scripts/cv_v2.py", "src/music_genre/evaluation/crossval.py")},
        "folds_sha256": sha256_file(ROOT / config["folds"]), "feature_cache": pointer["folder"],
        "feature_report_sha256": pointer["report_sha256"], "feature_dimensions": x.shape[1],
        "windows": len(x), "tracks": len(track_ids),
        "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn")},
        "outer_folds": records, "mean_fold_macro_f1": float(np.mean(fold_scores)),
        "std_fold_macro_f1": float(np.std(fold_scores, ddof=1)),
        "pooled_oof_macro_f1": macro_f1(tracks, y, oof),
        "plain_cv_macro_f1": {str(c): float(np.mean(s)) for c, s in plain.items()}, "final_c": final_c,
        "oof_tracks_sha256": sha256_file(folder / "oof_tracks.csv"),
        "elapsed_seconds": time.perf_counter() - started, "v2_test_audio_loaded": False,
    }
    write_json(folder / "report.json", result)
    print(json.dumps({k: result[k] for k in ("system", "mean_fold_macro_f1", "std_fold_macro_f1",
                                             "pooled_oof_macro_f1", "final_c", "elapsed_seconds")}, indent=1))


if __name__ == "__main__":
    main()
