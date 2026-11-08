"""T4: audio and cached validation checks for the six pinned linear systems."""

import json
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from check_candidate_inference import read_csv
from threadpoolctl import threadpool_limits

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.metrics import aggregate_tracks, classification_metrics
from music_genre.evaluation.window_geometry import window_sample_geometry
from music_genre.inference.adapters import OriginalBlock6Predictor
from music_genre.inference.linear_adapters import LinearComparatorPredictor
from music_genre.inference.predict import CandidatePredictor

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = ROOT / "reports/COMPARATOR_INFERENCE_PROGRESS.json"


def main():
    started = perf_counter()
    prep_path = ROOT / "reports/PREPARATION_EVALUATION_FINALE.json"
    prepared = json.loads(prep_path.read_text("utf-8"))
    protocol_path = ROOT / prepared["protocol_path"]
    if sha256_file(protocol_path) != prepared["protocol_sha256"]:
        raise ValueError("Protocol changed")
    protocol = json.loads(protocol_path.read_text("utf-8"))
    if protocol["test_evaluation_allowed"]:
        raise ValueError("Validation-only check")
    sample_path = ROOT / prepared["sample"]["path"]
    manifest_path = ROOT / prepared["baseline"]["folder"] / "manifest.csv"
    if (sha256_file(sample_path) != prepared["sample"]["sha256"]
            or sha256_file(manifest_path) != prepared["baseline"]["manifest_sha256"]):
        raise ValueError("Panel or manifest changed")
    cohort = read_csv(sample_path)
    manifest = {r["track_id"]: r for r in read_csv(manifest_path) if r["split"] == "validation"}
    if len(cohort) != 64 or len({r["track_id"] for r in cohort}) != 64 or len(manifest) != 780:
        raise ValueError("Invalid validation cohort")
    for row in cohort:
        if row["split"] != "validation" or any(manifest.get(row["track_id"], {}).get(k) != v for k, v in row.items()):
            raise ValueError("Panel differs from validation manifest")
    sources = ["scripts/check_comparator_inference.py", "scripts/check_candidate_inference.py",
               "src/music_genre/inference/linear_adapters.py", "src/music_genre/inference/adapters.py",
               "src/music_genre/inference/predict.py", "src/music_genre/audio/processing.py",
               "src/music_genre/data/audit.py", "src/music_genre/evaluation/metrics.py",
               "src/music_genre/evaluation/window_geometry.py",
               "src/music_genre/features/descriptors.py", "src/music_genre/features/vit.py",
               "src/music_genre/features/logmel.py", "src/music_genre/features/hpss.py",
               "src/music_genre/features/hpss_block6.py", "src/music_genre/models/vit.py",
               "src/music_genre/models/vit_channels.py", "src/music_genre/models/linear.py"]
    hashes = {p: sha256_file(ROOT / p) for p in sources}
    for p in (prep_path, protocol_path, sample_path, manifest_path):
        hashes[p.relative_to(ROOT).as_posix()] = sha256_file(p)
    for name, digest in prepared["checkpoint"]["sha256"].items():
        p = ROOT / "artifacts/pretrained/vit_tiny" / name
        if sha256_file(p) != digest:
            raise ValueError("Pretrained checkpoint changed")
        hashes[p.relative_to(ROOT).as_posix()] = digest
    labels = prepared["labels"]
    systems, common_index = {}, None
    for system in protocol["linear_systems"]:
        began = perf_counter()
        pin = prepared["runs"][system]
        run_path = ROOT / pin["run"]["folder"] / "report.json"
        cache = ROOT / pin["feature_cache"]
        feature_path = cache / "report.json"
        reference_path = ROOT / pin["run"]["folder"] / "validation_predictions.csv"
        for p, digest in ((run_path, pin["run"]["report_sha256"]),
                          (feature_path, pin["feature_report_sha256"]),
                          (reference_path, pin["validation_predictions_sha256"]),
                          (ROOT / pin["model_path"], pin["model_sha256"])):
            if sha256_file(p) != digest:
                raise ValueError(f"Changed export: {system}/{p.name}")
            hashes[p.relative_to(ROOT).as_posix()] = digest
        run = json.loads(run_path.read_text("utf-8"))
        features = json.loads(feature_path.read_text("utf-8"))
        if (run["test_loaded"] or run["test_evaluated"] or features["test_audio_loaded"]
                or features["provenance"]["splits"] != ["train", "validation"]
                or run["labels"] != labels or run["baseline_manifest_sha256"] != prepared["baseline"]["manifest_sha256"]):
            raise ValueError("Incompatible cache or run")
        for filename, field in (("index.csv", "index_sha256"), ("features.npy", "features_sha256")):
            p = cache / filename
            if sha256_file(p) != features[field]:
                raise ValueError("Cached data changed")
            hashes[p.relative_to(ROOT).as_posix()] = features[field]
        index = read_csv(cache / "index.csv")
        if common_index is None:
            common_index = index
        if index != common_index or any(int(r["row_index"]) != i or r["split"] not in ("train", "validation") for i, r in enumerate(index)):
            raise ValueError("Window alignment differs across comparators")
        positions = [i for i, r in enumerate(index) if r["split"] == "validation"]
        val_index = [index[i] for i in positions]
        for row in val_index:
            if row["track_id"] not in manifest or any(row[k] != manifest[row["track_id"]][k] for k in ("artist_id", "label_index", "source_sha256")):
                raise ValueError("Cached validation metadata mismatch")
        if system == "vit_block6":
            predictor = OriginalBlock6Predictor(ROOT)
        elif system == "vit_block6_hpss":
            predictor = CandidatePredictor(ROOT)
        else:
            predictor = LinearComparatorPredictor(ROOT, system)
        matrix = np.load(cache / "features.npy", mmap_mode="r", allow_pickle=False)
        if list(matrix.shape) != features["matrix_shape"] or matrix.shape[0] != len(index):
            raise ValueError("Invalid cached matrix dimensions")
        probabilities = predictor.model.predict_proba(matrix[positions])
        ids, truth, scores = aggregate_tracks(val_index, probabilities, len(labels))
        reference = read_csv(reference_path)
        if ids != [r["track_id"] for r in reference] or ids != sorted(manifest):
            raise ValueError("Track alignment mismatch")
        expected = np.array([[float(r[f"p_{label}"]) for label in labels] for r in reference])
        if any(r["true_label"] != manifest[r["track_id"]]["genre"] for r in reference):
            raise ValueError("Reference label mismatch")
        cache_error = float(np.max(np.abs(scores - expected)))
        metrics = classification_metrics(truth, scores, labels)
        if cache_error > 1e-6 or metrics != run["validation"]:
            raise ValueError("Complete validation metrics not reproduced from cache")
        lookup = {track: i for i, track in enumerate(ids)}
        windows_by_track = {}
        for i, row in enumerate(val_index):
            windows_by_track.setdefault(row["track_id"], []).append(i)
        results, track_error, window_error = [], 0., 0.
        for number, row in enumerate(cohort, 1):
            if perf_counter() - started > 1800:
                raise RuntimeError("Thirty-minute verification budget exceeded")
            path = (ROOT / "data/raw" / row["audio_path"]).resolve()
            if not path.is_relative_to((ROOT / "data/raw/fma_small").resolve()) or sha256_file(path) != row["source_sha256"]:
                raise ValueError("Source audio changed")
            with threadpool_limits(limits=1):
                prediction = predictor.predict_file(path)
            group = windows_by_track[row["track_id"]]
            geometry = window_sample_geometry([val_index[i] for i in group], 16000)
            if (geometry != window_sample_geometry(prediction["windows"], 16000)
                    or prediction["labels"] != labels):
                raise ValueError("Window geometry or labels differ")
            error = float(np.max(np.abs(np.array(prediction["scores"]) - expected[lookup[row["track_id"]]])))
            window_diff = float(np.max(np.abs(np.array([w["scores"] for w in prediction["windows"]]) - probabilities[group])))
            if error > 1e-6 or window_diff > 1e-6:
                raise ValueError(f"Audio/cache mismatch: {system}/{row['track_id']}: {error}, windows {window_diff}")
            if sha256_file(path) != row["source_sha256"]:
                raise ValueError("Audio changed during inference")
            track_error, window_error = max(track_error, error), max(window_error, window_diff)
            results.append({"track_id": row["track_id"], "source_sha256": row["source_sha256"],
                            "track_probability_error": error, "window_probability_error": window_diff,
                            "prediction": prediction})
            if number % 16 == 0:
                state = {"status": "running", "system": system, "completed_tracks": number,
                         "completed_systems": list(systems), "max_track_error": track_error,
                         "max_window_error": window_error, "test_evaluated": False}
                write_json(PROGRESS, state)
                print(json.dumps(state), flush=True)
        systems[system] = {"model_id": predictor.model_id, "feature_dimensions": matrix.shape[1],
                           "panel_tracks": 64, "panel_windows": sum(len(r["prediction"]["windows"]) for r in results),
                           "max_track_probability_error": track_error, "max_window_probability_error": window_error,
                           "cache_probability_error": cache_error, "full_validation": metrics,
                           "elapsed_seconds": perf_counter() - began, "results": results}
    if any(sha256_file(ROOT / p) != digest for p, digest in hashes.items()):
        raise ValueError("Sources or inputs changed during verification")
    report = {"status": "complete", "test_evaluated": False, "test_audio_loaded": False,
              "systems": systems, "source_and_input_sha256": hashes, "tolerance": 1e-6,
              "elapsed_seconds": perf_counter() - started, "numerical_threads": 1,
              "versions": {p: version(p) for p in ("numpy", "torch", "timm", "librosa", "soundfile")}}
    destination = ROOT / "reports/COMPARATOR_INFERENCE_CHECK.json"
    write_json(destination, report)
    write_json(ROOT / "reports/latest_comparator_inference_check.json",
               {"report": destination.relative_to(ROOT).as_posix(), "report_sha256": sha256_file(destination)})
    write_json(PROGRESS, {"status": "complete", "systems": list(systems), "test_evaluated": False})
    print(json.dumps({"status": "complete", "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        write_json(PROGRESS, {"status": "failed", "error": str(error), "test_evaluated": False})
        raise
