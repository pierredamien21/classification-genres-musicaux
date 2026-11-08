"""T5: verify the three selected CNN exports on audio and cached validation."""

import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from check_candidate_inference import read_csv
from threadpoolctl import threadpool_limits

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.metrics import aggregate_tracks, classification_metrics
from music_genre.evaluation.window_geometry import window_sample_geometry
from music_genre.inference.cnn_adapter import CnnPredictor
from music_genre.training.cnn import predict_windows

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = ROOT / "reports/CNN_INFERENCE_PROGRESS.json"


def main():
    started = perf_counter()
    prep_path = ROOT / "reports/PREPARATION_EVALUATION_FINALE.json"
    prepared = json.loads(prep_path.read_text("utf-8"))
    protocol_path = ROOT / prepared["protocol_path"]
    if sha256_file(protocol_path) != prepared["protocol_sha256"]:
        raise ValueError("Protocol changed")
    protocol = json.loads(protocol_path.read_text("utf-8"))
    if protocol["test_evaluation_allowed"] or protocol["cnn_runs"] != [f"cnn20_seed{s}" for s in (42, 43, 44)]:
        raise ValueError("Unexpected or opened protocol")
    sample_path = ROOT / prepared["sample"]["path"]
    manifest_path = ROOT / prepared["baseline"]["folder"] / "manifest.csv"
    if (sha256_file(sample_path) != prepared["sample"]["sha256"]
            or sha256_file(manifest_path) != prepared["baseline"]["manifest_sha256"]):
        raise ValueError("Panel or manifest changed")
    cohort = read_csv(sample_path)
    manifest = {r["track_id"]: r for r in read_csv(manifest_path) if r["split"] == "validation"}
    if len(cohort) != 64 or len({r["track_id"] for r in cohort}) != 64 or len(manifest) != 780:
        raise ValueError("Unexpected validation cohort")
    for row in cohort:
        if row["split"] != "validation" or any(manifest.get(row["track_id"], {}).get(k) != v for k, v in row.items()):
            raise ValueError("Panel not in validation manifest")
    first_pin = prepared["runs"]["cnn20_seed42"]
    cache = ROOT / first_pin["feature_cache"]
    feature_path = cache / "report.json"
    if sha256_file(feature_path) != first_pin["feature_report_sha256"]:
        raise ValueError("Feature report changed")
    features = json.loads(feature_path.read_text("utf-8"))
    if features["test_audio_loaded"] or features["provenance"]["splits"] != ["train", "validation"]:
        raise ValueError("Cache includes test")
    sources = ["scripts/check_cnn_inference.py", "scripts/check_candidate_inference.py",
               "src/music_genre/inference/cnn_adapter.py", "src/music_genre/inference/predict.py",
               "src/music_genre/audio/processing.py", "src/music_genre/features/logmel.py",
               "src/music_genre/models/cnn.py", "src/music_genre/training/cnn.py",
               "src/music_genre/evaluation/metrics.py", "src/music_genre/evaluation/window_geometry.py",
               "src/music_genre/data/audit.py"]
    hashes = {p: sha256_file(ROOT / p) for p in sources}
    for path in (prep_path, protocol_path, sample_path, manifest_path, feature_path):
        hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    for name, field in (("index.csv", "index_sha256"), ("features.npy", "features_sha256")):
        path = cache / name
        if sha256_file(path) != features[field]:
            raise ValueError("Cached artifact changed")
        hashes[path.relative_to(ROOT).as_posix()] = features[field]
    index = read_csv(cache / "index.csv")
    if any(int(r["row_index"]) != i or r["split"] not in ("train", "validation") for i, r in enumerate(index)):
        raise ValueError("Invalid cache row order or split")
    positions = np.array([i for i, r in enumerate(index) if r["split"] == "validation"])
    val_index = [index[i] for i in positions]
    for row in val_index:
        if row["track_id"] not in manifest or any(row[k] != manifest[row["track_id"]][k] for k in ("artist_id", "label_index", "source_sha256")):
            raise ValueError("Cache metadata differs from manifest")
    matrix = np.load(cache / "features.npy", mmap_mode="r", allow_pickle=False)
    if matrix.shape != (len(index), 128, 622) or matrix.dtype != np.float32:
        raise ValueError("Unexpected native CNN input")
    labels = prepared["labels"]
    systems = {}
    for seed in (42, 43, 44):
        name = f"cnn20_seed{seed}"
        pin = prepared["runs"][name]
        if pin["feature_cache"] != first_pin["feature_cache"] or pin["feature_report_sha256"] != first_pin["feature_report_sha256"]:
            raise ValueError("CNN caches differ")
        predictor = CnnPredictor(ROOT, seed)
        run_path = ROOT / pin["run"]["folder"] / "report.json"
        run = json.loads(run_path.read_text("utf-8"))
        if run["test_loaded"] or run["test_evaluated"] or run["baseline_manifest_sha256"] != prepared["baseline"]["manifest_sha256"]:
            raise ValueError("CNN run is not compatible")
        reference_path = ROOT / pin["run"]["folder"] / "validation_predictions.csv"
        if sha256_file(reference_path) != pin["validation_predictions_sha256"]:
            raise ValueError("Reference predictions changed")
        for path in (run_path, reference_path, ROOT / pin["model_path"]):
            hashes[path.relative_to(ROOT).as_posix()] = sha256_file(path)
        reference = read_csv(reference_path)
        if [(r["track_id"], r["true_label"]) for r in reference] != [(t, manifest[t]["genre"]) for t in sorted(manifest)]:
            raise ValueError("Reference alignment differs")
        began = perf_counter()
        write_json(PROGRESS, {"status": "running", "system": name, "phase": "cached_validation",
                              "completed_systems": list(systems), "test_evaluated": False})
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(predictor.config["threads"])
            with threadpool_limits(limits=predictor.config["threads"]):
                cached_windows = predict_windows(predictor.model.native, matrix, positions, predictor.config)
        finally:
            torch.set_num_threads(previous_threads)
        ids, truth, cached_scores = aggregate_tracks(val_index, cached_windows, len(labels))
        expected = np.array([[float(r[f"p_{label}"]) for label in labels] for r in reference])
        cache_error = float(np.max(np.abs(cached_scores - expected)))
        metrics = classification_metrics(truth, cached_scores, labels)
        if ids != [r["track_id"] for r in reference] or cache_error > 1e-6 or metrics != run["validation"]:
            raise ValueError(f"Cached validation not reproduced: {name}, error {cache_error}")
        lookup = {track: i for i, track in enumerate(ids)}
        groups = {}
        for i, row in enumerate(val_index):
            groups.setdefault(row["track_id"], []).append(i)
        results, largest, largest_window = [], 0., 0.
        for number, row in enumerate(cohort, 1):
            if perf_counter() - started > 1800:
                raise RuntimeError("Thirty-minute verification budget exceeded")
            path = (ROOT / "data/raw" / row["audio_path"]).resolve()
            if not path.is_relative_to((ROOT / "data/raw/fma_small").resolve()) or sha256_file(path) != row["source_sha256"]:
                raise ValueError("Raw audio changed")
            result = predictor.predict_file(path)
            group = groups[row["track_id"]]
            if (result["labels"] != labels or window_sample_geometry([val_index[i] for i in group], 16000)
                    != window_sample_geometry(result["windows"], 16000)):
                raise ValueError("Window geometry or labels differ")
            error = float(np.max(np.abs(np.array(result["scores"]) - expected[lookup[row["track_id"]]])))
            window_error = float(np.max(np.abs(np.array([w["scores"] for w in result["windows"]]) - cached_windows[group])))
            if error > 1e-6:
                raise ValueError(f"Audio predictions differ: {name}/{row['track_id']}, {error}")
            if sha256_file(path) != row["source_sha256"]:
                raise ValueError("Audio changed during inference")
            results.append({"track_id": row["track_id"], "source_sha256": row["source_sha256"],
                            "track_probability_error": error, "window_probability_error": window_error,
                            "prediction": result})
            largest, largest_window = max(largest, error), max(largest_window, window_error)
            if number % 16 == 0:
                state = {"status": "running", "system": name, "phase": "audio_panel", "completed_tracks": number,
                         "completed_systems": list(systems), "max_track_error": largest,
                         "max_window_error": largest_window, "test_evaluated": False}
                write_json(PROGRESS, state)
                print(json.dumps(state), flush=True)
        systems[name] = {"seed": seed, "selected_epoch": predictor.selected_epoch, "model_id": predictor.model_id,
                         "panel_tracks": 64, "panel_windows": 192, "max_track_probability_error": largest,
                         "max_window_probability_error": largest_window, "cache_probability_error": cache_error,
                         "full_validation": metrics, "elapsed_seconds": perf_counter() - began, "results": results}
    if any(sha256_file(ROOT / p) != digest for p, digest in hashes.items()):
        raise ValueError("Sources or inputs changed during run")
    averages = {metric: {"mean": float(np.mean([s["full_validation"][metric] for s in systems.values()])),
                         "sample_std": float(np.std([s["full_validation"][metric] for s in systems.values()], ddof=1))}
                for metric in ("macro_f1", "accuracy")}
    report = {"status": "complete", "systems": systems, "mean_of_metrics": averages,
              "aggregation": "mean_of_per_seed_metrics_not_probability_ensemble",
              "source_and_input_sha256": hashes, "track_tolerance": 1e-6,
              "window_errors": "reported_separately_native_batch_size_16_vs_single_track_batch_up_to_3",
              "test_evaluated": False, "test_audio_loaded": False, "elapsed_seconds": perf_counter() - started,
              "versions": {p: version(p) for p in ("torch", "numpy", "librosa", "soundfile", "safetensors")}}
    destination = ROOT / "reports/CNN_INFERENCE_CHECK.json"
    write_json(destination, report)
    write_json(ROOT / "reports/latest_cnn_inference_check.json",
               {"report": destination.relative_to(ROOT).as_posix(), "report_sha256": sha256_file(destination)})
    write_json(PROGRESS, {"status": "complete", "systems": list(systems), "test_evaluated": False})
    print(json.dumps({"status": "complete", "elapsed_seconds": report["elapsed_seconds"], "mean_of_metrics": averages}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(PROGRESS, {"status": "failed", "error": str(error), "test_evaluated": False})
        raise
