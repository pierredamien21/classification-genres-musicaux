"""Fit training windows, choose C on validation tracks, leave test unopened."""

import argparse
import csv
import json
import time
import warnings
from collections import Counter
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.metrics import aggregate_tracks, classification_metrics
from music_genre.features.windows import central_window_rows
from music_genre.models.linear import LinearProbabilityModel

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", choices=(
        "descriptors", "vit", "vit_central", "vit_block6", "vit_hpss", "vit_block6_hpss",
        "fusion", "fusion_no_rhythm", "fusion_no_harmony",
    ), default="descriptors")
    args = parser.parse_args()
    config_path = ROOT / f"configs/baseline_{args.features}.json"
    config = json.loads(config_path.read_text())
    pointer = json.loads((ROOT / f"data/features/latest_{args.features}.json").read_text())
    cache = (ROOT / pointer["folder"]).resolve()
    if not cache.is_relative_to((ROOT / "data/features").resolve()):
        raise ValueError("Cache outside feature directory")
    if sha256_file(cache / "report.json") != pointer["report_sha256"]:
        raise ValueError("Feature report changed")
    feature_report = json.loads((cache / "report.json").read_text(encoding="utf-8"))
    for filename, checksum in (("features.npy", "features_sha256"), ("index.csv", "index_sha256")):
        if sha256_file(cache / filename) != feature_report[checksum]:
            raise ValueError(f"Feature artifact changed: {filename}")
    baseline = json.loads((ROOT / "data/manifests/latest_baseline.json").read_text())
    manifest = (ROOT / baseline["folder"] / "manifest.csv").resolve()
    if not manifest.is_relative_to((ROOT / "data/manifests").resolve()):
        raise ValueError("Manifest outside data directory")
    if (sha256_file(manifest) != baseline["manifest_sha256"]
            or baseline["manifest_sha256"] != feature_report["provenance"]["manifest_sha256"]):
        raise ValueError("Features belong to another baseline selection")
    with manifest.open(encoding="utf-8", newline="") as stream:
        selected = {r["track_id"]: r for r in csv.DictReader(stream)
                    if r["split"] in ("train", "validation")}
    with (cache / "index.csv").open(encoding="utf-8", newline="") as stream:
        index = list(csv.DictReader(stream))
    if args.features != "descriptors":
        descriptor_pointer = json.loads((ROOT / "data/features/latest_descriptors.json").read_text())
        descriptor_cache = (ROOT / descriptor_pointer["folder"]).resolve()
        if not descriptor_cache.is_relative_to((ROOT / "data/features").resolve()):
            raise ValueError("Descriptor cache outside feature directory")
        if sha256_file(descriptor_cache / "report.json") != descriptor_pointer["report_sha256"]:
            raise ValueError("Descriptor report changed")
        descriptor_report = json.loads((descriptor_cache / "report.json").read_text())
        if sha256_file(descriptor_cache / "index.csv") != descriptor_report["index_sha256"]:
            raise ValueError("Descriptor index changed")
        if args.features == "vit_central":
            with (descriptor_cache / "index.csv").open(encoding="utf-8", newline="") as stream:
                _, expected = central_window_rows(list(csv.DictReader(stream)))
            if index != expected or feature_report["provenance"]["config"]["offsets_seconds"] != [10.0]:
                raise ValueError("Central features must be the exact reference subset")
        elif descriptor_report["index_sha256"] != feature_report["index_sha256"]:
            raise ValueError("ViT and descriptors must use identical indexed windows")
    x = np.load(cache / "features.npy", allow_pickle=False).astype(np.float64)
    labels = feature_report["provenance"]["labels"]
    if labels != json.loads((ROOT / "configs/labels.json").read_text())["labels"]:
        raise ValueError("Class order changed")
    if (x.shape != (len(index), len(feature_report["feature_names"]))
            or not np.isfinite(x).all() or {r["track_id"] for r in index} != set(selected)):
        raise ValueError("Incomplete or invalid features")
    for position, row in enumerate(index):
        source = selected[row["track_id"]]
        if (int(row["row_index"]) != position or row["split"] not in ("train", "validation")
                or any(row[key] != source[key] for key in
                       ("split", "label_index", "artist_id", "source_sha256"))):
            raise ValueError("Feature index does not match its manifest")
    train_positions = [i for i, r in enumerate(index) if r["split"] == "train"]
    val_positions = [i for i, r in enumerate(index) if r["split"] == "validation"]
    train_index = [index[i] for i in train_positions]
    val_index = [index[i] for i in val_positions]
    train_artists = {r["artist_id"] for r in train_index}
    if train_artists & {r["artist_id"] for r in val_index}:
        raise ValueError("Artist overlap between train and validation")
    x_train, x_val = x[train_positions], x[val_positions]
    y_train = np.array([int(r["label_index"]) for r in train_index])
    if set(y_train) != set(range(len(labels))):
        raise ValueError("Training classes incomplete")
    folder = ROOT / "runs" / (args.features + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "config.json", config)
    write_json(folder / "feature_config.json", feature_report["provenance"]["config"])
    started = time.perf_counter()
    trials = []
    best = None
    with threadpool_limits(limits=4):
        for c in sorted(config["c_grid"]):
            pipeline = Pipeline([
                ("scale", StandardScaler()),
                ("classifier", LogisticRegression(
                    C=c, solver=config["solver"], max_iter=config["max_iter"],
                    tol=config["tolerance"], random_state=config["seed"],
                    class_weight=config["class_weight"],
                )),
            ])
            fit_started = time.perf_counter()
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always")
                pipeline.fit(x_train, y_train)
            if any(issubclass(w.category, ConvergenceWarning) for w in captured):
                raise RuntimeError(f"Model C={c} did not converge; do not publish an incomplete fit")
            classes = pipeline.named_steps["classifier"].classes_
            if not np.array_equal(classes, np.arange(len(labels))):
                raise ValueError("Unexpected classifier class order")
            window_probabilities = pipeline.predict_proba(x_val)
            track_ids, truth, probabilities = aggregate_tracks(val_index, window_probabilities, len(labels))
            metrics = classification_metrics(truth, probabilities, labels)
            trial = {
                "c": c, "fit_and_validation_seconds": time.perf_counter() - fit_started,
                "iterations": pipeline.named_steps["classifier"].n_iter_.tolist(),
                "validation": metrics, "warnings": [str(w.message) for w in captured],
            }
            trials.append(trial)
            print(f"C={c}: validation macro-F1={metrics['macro_f1']:.4f}, "
                  f"accuracy={metrics['accuracy']:.4f}", flush=True)
            if best is None or metrics["macro_f1"] > best[0]["validation"]["macro_f1"]:
                best = (trial, pipeline, track_ids, truth, probabilities)
        trial, pipeline, track_ids, truth, probabilities = best
        train_ids, train_truth, train_probabilities = aggregate_tracks(
            train_index, pipeline.predict_proba(x_train), len(labels)
        )
    scaler, classifier = pipeline.named_steps["scale"], pipeline.named_steps["classifier"]
    if (scaler.n_samples_seen_ != len(x_train)
            or not np.allclose(scaler.mean_, x_train.mean(axis=0), atol=1e-10)):
        raise ValueError("Scaler was not fitted exclusively on the training windows")
    model = LinearProbabilityModel(
        scaler.mean_, scaler.scale_, classifier.coef_, classifier.intercept_, classifier.classes_
    )
    model.save(folder / "model.npz")
    restored = LinearProbabilityModel.load(folder / "model.npz")
    restored_probabilities = restored.predict_proba(x_val)
    max_error = float(np.max(np.abs(restored_probabilities - pipeline.predict_proba(x_val))))
    if max_error > 1e-10:
        raise ValueError(f"Export mismatch: {max_error}")
    majority = int(np.bincount(train_truth, minlength=len(labels)).argmax())
    majority_proba = np.zeros((len(truth), len(labels)))
    majority_proba[:, majority] = 1
    write_json(folder / "model.json", {
        "schema_version": 1, "family": "standardized_multinomial_logistic_regression",
        "labels": labels, "feature_names": feature_report["feature_names"],
        "feature_config": feature_report["provenance"]["config"], "selected_c": trial["c"],
        "model_sha256": sha256_file(folder / "model.npz"),
        "aggregation": "mean_window_probabilities", "test_evaluated": False,
    })
    prediction_rows = [
        {"track_id": key, "true_label": labels[int(y)], "predicted_label": labels[int(p.argmax())],
         **{f"p_{label}": float(p[i]) for i, label in enumerate(labels)}}
        for key, y, p in zip(track_ids, truth, probabilities, strict=True)
    ]
    with (folder / "validation_predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(prediction_rows[0]))
        writer.writeheader()
        writer.writerows(prediction_rows)
    # Feature ranges are calculated from train, not from the held-out test set.
    feature_stats = [
        {"name": name, "family": feature_report["feature_families"][i],
         "train_mean": float(scaler.mean_[i]), "train_std": float(np.sqrt(scaler.var_[i])),
         "train_min": float(x_train[:, i].min()), "train_max": float(x_train[:, i].max())}
        for i, name in enumerate(feature_report["feature_names"])
    ]
    write_json(folder / "train_feature_statistics.json", feature_stats)
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(),
        "config": config, "config_sha256": sha256_file(config_path),
        "versions": {name: version(name) for name in ("numpy", "scipy", "scikit-learn")},
        "feature_cache": pointer["folder"], "feature_report_sha256": pointer["report_sha256"],
        "baseline_manifest_sha256": baseline["manifest_sha256"],
        "labels": labels, "window_counts": {"train": len(train_index), "validation": len(val_index)},
        "track_counts": {"train": len(train_ids), "validation": len(track_ids)},
        "feature_dimensions": x.shape[1],
        "family_dimensions": dict(Counter(feature_report["feature_families"])),
        "trials": trials, "selected_c": trial["c"], "validation": trial["validation"],
        "training": classification_metrics(train_truth, train_probabilities, labels),
        "majority_class": labels[majority],
        "majority_validation": classification_metrics(truth, majority_proba, labels),
        "elapsed_seconds": time.perf_counter() - started,
        "export_max_probability_error": max_error,
        "model_sha256": sha256_file(folder / "model.npz"),
        "validation_predictions_sha256": sha256_file(folder / "validation_predictions.csv"),
        "source_sha256": {name: sha256_file(ROOT / name) for name in (
            "scripts/train_descriptor_baseline.py", "src/music_genre/models/linear.py",
            "src/music_genre/evaluation/metrics.py",
        )},
        "test_loaded": False, "test_evaluated": False,
        "limitations": [
            "Validation is used to choose C; its reported score is not an unbiased final estimate.",
            "Filtered FMA-small selection; no direct comparison with unfiltered benchmark scores.",
            "Linear classifier on frozen representations; no fine-tuning or CNN in this run.",
            "Representation-specific assumptions are described in the feature configuration.",
            "No test-set confidence interval or calibration claim.",
        ],
    }
    write_json(folder / "report.json", report)
    pointer_name = "descriptor" if args.features == "descriptors" else args.features
    write_json(ROOT / f"reports/latest_{pointer_name}_run.json",
               {"folder": folder.relative_to(ROOT).as_posix(),
                "report_sha256": sha256_file(folder / "report.json")})
    print(f"Selected C={trial['c']}; report: {folder / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
