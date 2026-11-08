"""T2b: compare the pinned original block six with the completed HPSS panel."""

import hashlib
import json
import tempfile
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
import soundfile as sf
from check_candidate_inference import read_csv

from music_genre.audio.processing import load_audio
from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.errors import paired_outcomes
from music_genre.evaluation.robustness import compare_condition, perturb_audio
from music_genre.inference.adapters import OriginalBlock6Predictor

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = ROOT / "reports/BLOCK6_ROBUSTNESS_PROGRESS.json"
BUDGET = 1800


def window_geometry(prediction):
    return [(w["start_seconds"], w["valid_seconds"], w["padded_seconds"])
            for w in prediction["windows"]]


def main():
    started = perf_counter()
    prepared_path = ROOT / "reports/PREPARATION_EVALUATION_FINALE.json"
    prepared = json.loads(prepared_path.read_text("utf-8"))
    pointer = json.loads((ROOT / "reports/latest_robustness_panel.json").read_text("utf-8"))
    hpss_path = ROOT / pointer["report"]
    if sha256_file(hpss_path) != pointer["report_sha256"]:
        raise ValueError("HPSS report changed")
    hpss = json.loads(hpss_path.read_text("utf-8"))
    hashes = dict(hpss["source_and_input_sha256"])
    if (hpss["status"] != "complete" or hpss["tracks"] != 64
            or hpss["test_evaluated"] or hpss["test_audio_loaded"]):
        raise ValueError("HPSS prerequisite incomplete")
    for path, digest in hashes.items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f"HPSS input changed: {path}")
    protocol_path = ROOT / prepared["protocol_path"]
    if sha256_file(protocol_path) != prepared["protocol_sha256"]:
        raise ValueError("Protocol changed")
    protocol = json.loads(protocol_path.read_text("utf-8"))
    settings = protocol["robustness"]
    conditions = ["identity", "gain_minus_6_db", "white_noise_snr_20_db"]
    if protocol["test_evaluation_allowed"] or settings["conditions"] != conditions:
        raise ValueError("Unexpected or opened protocol")
    sample_path = ROOT / prepared["sample"]["path"]
    manifest_path = ROOT / prepared["baseline"]["folder"] / "manifest.csv"
    cohort = read_csv(sample_path)
    validation = {r["track_id"]: r for r in read_csv(manifest_path) if r["split"] == "validation"}
    if (sha256_file(sample_path) != prepared["sample"]["sha256"]
            or sha256_file(manifest_path) != prepared["baseline"]["manifest_sha256"]
            or len(cohort) != 64 or len({r["track_id"] for r in cohort}) != 64
            or len({r["artist_id"] for r in cohort}) != 64):
        raise ValueError("Fixed panel changed")
    for row in cohort:
        if row["split"] != "validation" or any(validation.get(row["track_id"], {}).get(k) != v for k, v in row.items()):
            raise ValueError("Panel is not validation")
    old = {r["track_id"]: r for r in hpss["results"]}
    if list(old) != [r["track_id"] for r in cohort]:
        raise ValueError("HPSS panel alignment changed")
    pin = prepared["runs"]["vit_block6"]
    reference_path = ROOT / pin["run"]["folder"] / "validation_predictions.csv"
    if sha256_file(reference_path) != pin["validation_predictions_sha256"]:
        raise ValueError("Original reference predictions changed")
    reference = {r["track_id"]: r for r in read_csv(reference_path)}
    predictor = OriginalBlock6Predictor(ROOT)
    labels = predictor.labels
    if any(sum(r["genre"] == label for r in cohort) != 8 for label in labels):
        raise ValueError("Genre quotas changed")
    paths = [prepared_path, protocol_path, hpss_path, sample_path, manifest_path, reference_path,
             ROOT / pin["model_path"], ROOT / pin["run"]["folder"] / "report.json",
             ROOT / pin["feature_cache"] / "report.json"]
    paths += [ROOT / predictor.config["checkpoint_directory"] / name
              for name in predictor.config["checkpoint"]["sha256"]]
    paths += [ROOT / p for p in ("scripts/compare_block6_robustness.py",
              "src/music_genre/inference/adapters.py", "src/music_genre/features/vit.py",
              "src/music_genre/features/logmel.py", "src/music_genre/models/vit.py")]
    hashes.update({p.relative_to(ROOT).as_posix(): sha256_file(p) for p in paths})
    attempt = ROOT / "data/interim" / ("block6-robustness-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    attempt.mkdir(parents=True)
    pilot_ids = hpss["benchmark"]["track_ids"]
    ordered = [next(r for r in cohort if r["track_id"] == track) for track in pilot_ids]
    ordered += [r for r in cohort if r["track_id"] not in pilot_ids]
    results, benchmark = {}, None
    compute_started = perf_counter()
    with tempfile.TemporaryDirectory(dir=attempt, prefix="audio-") as temporary:
        wav = Path(temporary) / "transformed.wav"
        for number, row in enumerate(ordered, 1):
            path = (ROOT / "data/raw" / row["audio_path"]).resolve()
            if not path.is_relative_to((ROOT / "data/raw/fma_small").resolve()) or sha256_file(path) != row["source_sha256"]:
                raise ValueError("Raw audio changed")
            original = load_audio(path, sample_rate=16000).samples
            track = {k: row[k] for k in ("track_id", "artist_id", "genre", "source_sha256")}
            track["conditions"] = {}
            for condition in conditions:
                if perf_counter() - started > BUDGET:
                    raise RuntimeError("Thirty-minute budget exceeded")
                transformed = perturb_audio(original, condition, row["track_id"], seed=settings["seed"])
                sf.write(wav, transformed, 16000, subtype="FLOAT")
                decoded, rate = sf.read(wav, dtype="float32")
                if rate != 16000 or not np.array_equal(decoded, transformed):
                    raise ValueError("FLOAT WAV roundtrip changed audio")
                expected_audio = old[row["track_id"]]["conditions"][condition]["audio"]
                if (len(transformed) != expected_audio["samples"]
                        or float(np.abs(transformed).max()) != expected_audio["peak_abs"]):
                    raise ValueError("Signal differs from HPSS conditions")
                prediction = predictor.predict_file(wav, mode="fma")
                previous = old[row["track_id"]]["conditions"][condition]["prediction"]
                if (prediction["labels"] != previous["labels"]
                        or window_geometry(prediction) != window_geometry(previous)):
                    raise ValueError("Class order or windows differ from HPSS")
                info = {"pcm_sha256": hashlib.sha256(transformed.tobytes()).hexdigest()}
                if condition == "identity":
                    expected = [float(reference[row["track_id"]][f"p_{label}"]) for label in labels]
                    error = float(np.max(np.abs(np.array(expected) - prediction["scores"])))
                    info["saved_prediction_error"] = error
                    if error > 1e-6:
                        raise ValueError("Original clean predictions not reproduced")
                track["conditions"][condition] = {"audio": info, "prediction": prediction}
            if sha256_file(path) != row["source_sha256"]:
                raise ValueError("Raw audio changed during run")
            results[row["track_id"]] = track
            write_json(attempt / f"{row['track_id']}.json", track)
            if number == 8:
                elapsed = perf_counter() - compute_started
                benchmark = {"track_ids": pilot_ids, "elapsed_seconds": elapsed,
                             "projected_panel_seconds": elapsed * 8, "budget_seconds": BUDGET,
                             "workers": 1, "results_reused": True}
                write_json(attempt / "benchmark.json", benchmark)
                print(json.dumps({"benchmark": benchmark}), flush=True)
                if elapsed * 8 > BUDGET:
                    raise RuntimeError("Pilot exceeds thirty-minute budget")
            state = {"status": "running", "completed_tracks": number, "total_tracks": 64,
                     "elapsed_seconds": perf_counter() - started, "benchmark": benchmark,
                     "attempt": attempt.relative_to(ROOT).as_posix(), "test_evaluated": False}
            write_json(PROGRESS, state)
            if number % 8 == 0:
                print(json.dumps(state), flush=True)
    if any(sha256_file(ROOT / p) != digest for p, digest in hashes.items()):
        raise ValueError("Pinned sources or inputs changed during run")
    truth = np.array([labels.index(r["genre"]) for r in cohort])
    summaries = {}
    for name, collection in (("original", results), ("hpss", old)):
        matrices = {c: np.array([collection[r["track_id"]]["conditions"][c]["prediction"]["scores"] for r in cohort]) for c in conditions}
        summaries[name] = {}
        for condition in conditions:
            item = compare_condition(truth, matrices["identity"], matrices[condition], labels, settings["review_triggers"])
            item["paired_outcomes"] = paired_outcomes(truth, matrices["identity"].argmax(1), matrices[condition].argmax(1))
            item["flips_by_genre"] = {label: int(((truth == i) & (matrices["identity"].argmax(1) != matrices[condition].argmax(1))).sum()) for i, label in enumerate(labels)}
            summaries[name][condition] = item
        if name == "hpss" and summaries[name] != hpss["summary"]:
            raise ValueError("HPSS metrics not reproduced")
    report = {"status": "complete", "test_evaluated": False, "test_audio_loaded": False,
              "tracks": 64, "labels": labels, "conditions": conditions,
              "source_and_input_sha256": hashes, "model_id": predictor.model_id,
              "hpss_model_id": hpss["model_id"], "benchmark": benchmark,
              "elapsed_seconds": perf_counter() - started, "summary": summaries,
              "versions": {p: version(p) for p in ("numpy", "torch", "timm", "librosa", "soundfile")},
              "max_identity_probability_error": max(r["conditions"]["identity"]["audio"]["saved_prediction_error"] for r in results.values()),
              "results": [results[r["track_id"]] for r in cohort]}
    destination = ROOT / "reports/BLOCK6_ROBUSTNESS_COMPARISON.json"
    write_json(destination, report)
    write_json(ROOT / "reports/latest_block6_robustness_comparison.json",
               {"report": destination.relative_to(ROOT).as_posix(), "report_sha256": sha256_file(destination)})
    write_json(PROGRESS, {**state, "status": "complete", "elapsed_seconds": report["elapsed_seconds"]})
    print(json.dumps({"status": "complete", "max_identity_probability_error": report["max_identity_probability_error"],
                      "metrics": {m: {c: {k: s[k] for k in ("macro_f1_drop", "flipped_tracks")} for c, s in values.items()} for m, values in summaries.items()}}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        state = json.loads(PROGRESS.read_text("utf-8")) if PROGRESS.exists() else {}
        write_json(PROGRESS, {**state, "status": "failed", "error": str(error), "test_evaluated": False})
        raise
