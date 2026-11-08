"""Exercise the frozen bootstrap protocol on saved validation predictions only."""

import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from check_candidate_inference import read_csv

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.bootstrap import paired_artist_bootstrap
from music_genre.evaluation.metrics import classification_metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    started = perf_counter()
    preparation_path = ROOT / "reports/PREPARATION_EVALUATION_FINALE.json"
    prepared = json.loads(preparation_path.read_text("utf-8"))
    protocol_path = ROOT / prepared["protocol_path"]
    if sha256_file(protocol_path) != prepared["protocol_sha256"]:
        raise ValueError("Prepared protocol changed")
    protocol = json.loads(protocol_path.read_text("utf-8"))
    expected = {"unit": "artist", "resampling": "all_tracks_of_each_sampled_artist_with_multiplicity",
                "paired_across_systems": True, "replicates": 2000, "seed": 20261028,
                "generator": "numpy_PCG64", "confidence_level": .95, "interval": "percentile",
                "quantile_method": "linear", "resample_tracks_within_artist": False,
                "stratify_by_genre": False,
                "missing_class_replicates": "retain_fixed_labels_and_report_count",
                "claim": "approximate_pointwise_intervals_conditional_on_fixed_models_not_simultaneous_tests"}
    if protocol["test_evaluation_allowed"] or protocol["bootstrap"] != expected:
        raise ValueError("Unexpected bootstrap settings or test permitted")
    manifest_path = ROOT / prepared["baseline"]["folder"] / "manifest.csv"
    if sha256_file(manifest_path) != prepared["baseline"]["manifest_sha256"]:
        raise ValueError("Manifest changed")
    manifest = read_csv(manifest_path)
    validation = sorted([r for r in manifest if r["split"] == "validation"], key=lambda r: r["track_id"])
    if len(validation) != 780 or len({r["track_id"] for r in validation}) != 780:
        raise ValueError("Unexpected validation cohort")
    artist_sets = {s: {r["artist_id"] for r in manifest if r["split"] == s} for s in ("train", "validation", "test")}
    if any(artist_sets[a] & artist_sets[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("Artist partitions overlap")
    labels = prepared["labels"]
    if len(labels) != 8 or len(set(labels)) != 8:
        raise ValueError("Expected eight labels")
    if any(int(r["label_index"]) != labels.index(r["genre"]) for r in manifest):
        raise ValueError("Manifest label order mismatch")
    tracks = [{"track_id": r["track_id"], "artist_id": r["artist_id"],
               "label_index": int(r["label_index"])} for r in validation]
    inputs = [preparation_path, protocol_path, manifest_path, Path(__file__),
              ROOT / "src/music_genre/evaluation/bootstrap.py",
              ROOT / "src/music_genre/evaluation/metrics.py",
              ROOT / "scripts/check_candidate_inference.py", ROOT / "src/music_genre/data/audit.py"]
    predictions, references = {}, {}
    for name in protocol["linear_systems"] + protocol["cnn_runs"]:
        pin = prepared["runs"][name]
        folder = (ROOT / pin["run"]["folder"]).resolve()
        if not folder.is_relative_to((ROOT / "runs").resolve()):
            raise ValueError("Run outside workspace runs")
        run_path = folder / "report.json"
        prediction_path = folder / "validation_predictions.csv"
        if (sha256_file(run_path) != pin["run"]["report_sha256"]
                or sha256_file(prediction_path) != pin["validation_predictions_sha256"]
                or sha256_file(ROOT / pin["model_path"]) != pin["model_sha256"]):
            raise ValueError(f"Pinned run changed: {name}")
        run = json.loads(run_path.read_text("utf-8"))
        if (run["test_loaded"] or run["test_evaluated"] or run["labels"] != labels
                or run["baseline_manifest_sha256"] != prepared["baseline"]["manifest_sha256"]
                or run["validation_predictions_sha256"] != pin["validation_predictions_sha256"]):
            raise ValueError("Incompatible validation run")
        rows = sorted(read_csv(prediction_path), key=lambda r: r["track_id"])
        if [(r["track_id"], r["true_label"]) for r in rows] != [(r["track_id"], r["genre"]) for r in validation]:
            raise ValueError("Prediction/manifest alignment differs")
        matrix = np.array([[float(r[f"p_{label}"]) for label in labels] for r in rows])
        metrics = classification_metrics(np.array([t["label_index"] for t in tracks]), matrix, labels)
        if metrics != run["validation"]:
            raise ValueError("Saved validation metrics not reproduced")
        predictions[name] = [{"track_id": r["track_id"], "scores": scores.tolist()}
                             for r, scores in zip(rows, matrix, strict=True)]
        references[name] = metrics
        inputs += [run_path, prediction_path, ROOT / pin["model_path"]]
    train_counts = np.bincount([int(r["label_index"]) for r in manifest if r["split"] == "train"], minlength=8)
    majority = int(train_counts.argmax())
    descriptor = json.loads((ROOT / prepared["runs"]["descriptor"]["run"]["folder"] / "report.json").read_text("utf-8"))
    if labels[majority] != descriptor["majority_class"]:
        raise ValueError("Train majority differs from original baseline")
    predictions["majority"] = [{"track_id": t["track_id"], "scores": np.eye(8)[majority].tolist()} for t in tracks]
    hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in inputs}
    comparisons = [protocol["primary_comparison"], *protocol["secondary_comparisons"]]
    groups = {"cnn20_mean": protocol["cnn_runs"]}
    result = paired_artist_bootstrap(tracks, predictions, labels, replicates=2000,
                                     seed=20261028, comparisons=comparisons, metric_groups=groups)
    for name, metrics in references.items():
        for metric in result["metrics"]:
            if abs(result["estimates"][name][metric]["point"] - metrics[metric]) > 1e-12:
                raise ValueError("Bootstrap point estimate differs from reference")
    arrays = {key: result.pop(key) for key in ("replicate_scores", "artist_draw_counts", "replicate_track_counts")}
    archive_path = ROOT / "reports/BOOTSTRAP_VALIDATION_DRAWS.npz"
    np.savez_compressed(archive_path, **arrays)
    if any(sha256_file(ROOT / path) != digest for path, digest in hashes.items()):
        raise ValueError("Sources/inputs changed during bootstrap")
    result.update({"status": "complete", "split": "validation", "tracks": len(tracks),
                   "artists": len(result["artist_ids"]), "labels": labels, "settings": expected,
                   "ordering": "lexicographic_track_and_artist_ids", "metric_groups": groups,
                   "comparisons": comparisons, "source_and_input_sha256": hashes,
                   "majority_train_class": labels[majority], "train_class_counts": train_counts.tolist(),
                   "sampled_track_count_range": [int(arrays["replicate_track_counts"].min()), int(arrays["replicate_track_counts"].max())],
                   "cnn_point_sample_std": {metric: float(np.std([result["estimates"][m][metric]["point"] for m in protocol["cnn_runs"]], ddof=1)) for metric in result["metrics"]},
                   "archive": {"path": archive_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(archive_path)},
                   "elapsed_seconds": perf_counter() - started,
                   "versions": {p: version(p) for p in ("numpy", "scikit-learn")},
                   "audio_loaded": False, "test_audio_loaded": False, "test_evaluated": False,
                   "limitations": ["validation_used_for_model_selection", "fixed_models_conditional_intervals",
                                   "pointwise_not_simultaneous", "no_training_or_seed_resampling"]})
    destination = ROOT / "reports/BOOTSTRAP_VALIDATION.json"
    write_json(destination, result)
    write_json(ROOT / "reports/latest_bootstrap_validation.json",
               {"report": destination.relative_to(ROOT).as_posix(), "report_sha256": sha256_file(destination)})
    print(json.dumps({"artists": result["artists"], "missing_class_replicates": result["missing_class_replicates"],
                      "elapsed_seconds": result["elapsed_seconds"], "differences": result["differences"]}), flush=True)


if __name__ == "__main__":
    main()
