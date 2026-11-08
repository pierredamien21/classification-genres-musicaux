"""Reproduce the fixed 64-track validation panel via the public inference API."""

import csv
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from music_genre.data.audit import sha256_file, write_json
from music_genre.inference.predict import CandidatePredictor

ROOT = Path(__file__).resolve().parents[1]


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    started = perf_counter()
    source_paths = ["scripts/check_candidate_inference.py", "src/music_genre/inference/predict.py",
                    "src/music_genre/audio/processing.py", "src/music_genre/features/hpss.py",
                    "src/music_genre/features/hpss_block6.py", "src/music_genre/models/linear.py",
                    "src/music_genre/models/vit.py", "src/music_genre/models/vit_channels.py"]
    hashes = {p: sha256_file(ROOT / p) for p in source_paths}
    preparation_path = ROOT / "reports/PREPARATION_EVALUATION_FINALE.json"
    preparation_sha = sha256_file(preparation_path)
    prepared = json.loads(preparation_path.read_text("utf-8"))
    protocol_path = ROOT / prepared["protocol_path"]
    if sha256_file(protocol_path) != prepared["protocol_sha256"]:
        raise ValueError("Protocol changed")
    protocol = json.loads(protocol_path.read_text("utf-8"))
    tolerance = protocol["robustness"]["hard_checks"]["maximum_identity_probability_error"]
    if protocol["test_evaluation_allowed"] or protocol["robustness"]["split"] != "validation":
        raise ValueError("This check is restricted to the prepared validation protocol")
    sample_path = ROOT / prepared["sample"]["path"]
    if sha256_file(sample_path) != prepared["sample"]["sha256"]:
        raise ValueError("Panel changed")
    cohort = read_csv(sample_path)
    baseline = prepared["baseline"]
    manifest_path = ROOT / baseline["folder"] / "manifest.csv"
    if sha256_file(manifest_path) != baseline["manifest_sha256"]:
        raise ValueError("Manifest changed")
    manifest = {r["track_id"]: r for r in read_csv(manifest_path) if r["split"] == "validation"}
    if (len(cohort) != 64 or len({r['track_id'] for r in cohort}) != 64
            or len({r['artist_id'] for r in cohort}) != 64):
        raise ValueError("Incomplete or duplicated panel")
    for row in cohort:
        if row["split"] != "validation" or any(manifest.get(row["track_id"], {}).get(k) != v for k, v in row.items()):
            raise ValueError("Panel differs from validation manifest")
    pin = prepared["runs"]["vit_block6_hpss"]
    folder = ROOT / pin["run"]["folder"]
    reference_path = folder / "validation_predictions.csv"
    if sha256_file(reference_path) != pin["validation_predictions_sha256"]:
        raise ValueError("Reference predictions changed")
    reference = {r["track_id"]: r for r in read_csv(reference_path)}
    feature_report = json.loads((ROOT / pin["feature_cache"] / "report.json").read_text("utf-8"))
    index_path = ROOT / pin["feature_cache"] / "index.csv"
    if sha256_file(index_path) != feature_report["index_sha256"]:
        raise ValueError("Window index changed")
    index = {}
    for row in read_csv(index_path):
        if row["split"] == "validation":
            index.setdefault(row["track_id"], []).append(row)
    predictor = CandidatePredictor(ROOT)
    results, largest = [], 0.0
    for number, row in enumerate(cohort, 1):
        path = (ROOT / "data/raw" / row["audio_path"]).resolve()
        if (not path.is_relative_to((ROOT / "data/raw/fma_small").resolve())
                or sha256_file(path) != row["source_sha256"]):
            raise ValueError("Source audio changed or escaped dataset")
        result = predictor.predict_file(path, mode="fma")
        ref = reference[row["track_id"]]
        if ref["true_label"] != row["genre"]:
            raise ValueError("Reference label differs from manifest")
        expected = np.array([float(ref[f"p_{label}"]) for label in predictor.labels])
        error = float(np.max(np.abs(np.array(result["scores"]) - expected)))
        probabilities = np.array([w["scores"] for w in result["windows"]])
        if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)
                or error > tolerance):
            raise ValueError(f"Inference differs for {row['track_id']}: {error}")
        windows = sorted(index[row["track_id"]], key=lambda r: int(r["window_id"]))
        if len(windows) != len(result["windows"]) or any(
                float(a["start_seconds"]) != b["start_seconds"]
                or float(a["valid_seconds"]) != b["valid_seconds"]
                for a, b in zip(windows, result["windows"], strict=True)):
            raise ValueError("Inference window boundaries differ from cache")
        if sha256_file(path) != row["source_sha256"]:
            raise ValueError("Audio changed during inference")
        largest = max(largest, error)
        results.append({"track_id": row["track_id"], "artist_id": row["artist_id"],
                        "source_sha256": row["source_sha256"], "max_probability_error": error,
                        "prediction": result})
        if number % 8 == 0:
            print(json.dumps({"completed": number, "total": 64, "max_probability_error": largest}), flush=True)
    if (sha256_file(preparation_path) != preparation_sha
            or any(sha256_file(ROOT / p) != digest for p, digest in hashes.items())):
        raise ValueError("Inputs or sources changed during verification")
    report = {"status": "complete", "tracks": len(results), "windows": sum(len(r["prediction"]["windows"]) for r in results),
              "test_evaluated": False, "test_audio_loaded": False, "model_id": predictor.model_id,
              "preparation_sha256": preparation_sha, "sample": prepared["sample"],
              "run": pin["run"], "source_sha256": hashes, "tolerance": tolerance,
              "max_probability_error": largest, "elapsed_seconds": perf_counter() - started,
              "results": results}
    report_path = ROOT / "reports/CANDIDATE_INFERENCE_CHECK.json"
    write_json(report_path, report)
    write_json(ROOT / "reports/latest_candidate_inference_check.json", {
        "report": report_path.relative_to(ROOT).as_posix(), "report_sha256": sha256_file(report_path)})
    print(json.dumps({"status": "complete", "tracks": len(results), "max_probability_error": largest,
                      "elapsed_seconds": report["elapsed_seconds"]}), flush=True)


if __name__ == "__main__":
    main()
