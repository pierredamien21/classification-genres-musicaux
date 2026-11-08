"""Replay V1 validation predictions with the current code and compare them to the stored ones.

Proof required by phase 1 of docs/PLAN_V2.md: the refactored code must reproduce the
9 pinned exports (model identity, track and window probabilities) within 1e-6.
Reads validation audio only; never touches the test split.
"""

import argparse
import csv
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from threadpoolctl import threadpool_limits

from music_genre.data.audit import sha256_file, write_json
from music_genre.inference.predict import SYSTEMS, load_predictor

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "data/interim/final-validation-v1"
TOLERANCE = 1e-6


def stored(system, track):
    record = json.loads((CAMPAIGN / "predictions" / system / f"{track}.json").read_text("utf-8"))
    return record["payload"]["prediction"]


def compare(expected, actual):
    """Largest absolute probability difference, or an identity mismatch message."""
    for key in ("schema_version", "model_id", "mode", "labels"):
        if expected[key] != actual[key]:
            return None, f"{key} differs"
    starts = [[w[k] for k in ("start_seconds", "valid_seconds", "padded_seconds")] for w in expected["windows"]]
    if starts != [[w[k] for k in ("start_seconds", "valid_seconds", "padded_seconds")] for w in actual["windows"]]:
        return None, "window geometry differs"
    pairs = [(expected["scores"], actual["scores"])] + [
        (e["scores"], a["scores"]) for e, a in zip(expected["windows"], actual["windows"], strict=True)]
    error = max(float(np.max(np.abs(np.asarray(e) - np.asarray(a)))) for e, a in pairs)
    if np.argmax(expected["scores"]) != np.argmax(actual["scores"]):
        return error, "predicted class differs"
    return error, None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--systems", nargs="+", choices=SYSTEMS, default=list(SYSTEMS))
    parser.add_argument("--per-genre", type=int, default=0,
                        help="Tracks per genre (first IDs in sorted order); 0 = all 780")
    parser.add_argument("--output", default="reports/V1_EQUIVALENCE.json")
    args = parser.parse_args()
    campaign = json.loads((CAMPAIGN / "campaign.json").read_text("utf-8"))
    if campaign["split"] != "validation":
        raise ValueError("Expected the V1 validation campaign")
    manifest = json.loads((ROOT / "data/manifests/latest_baseline.json").read_text("utf-8"))
    with (ROOT / manifest["folder"] / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = {r["track_id"]: r for r in csv.DictReader(stream)}
    tracks = sorted(campaign["track_ids"])
    if any(rows[t]["split"] != "validation" for t in tracks):
        raise ValueError("Non-validation track in campaign")
    if args.per_genre:
        picked, counts = [], {}
        for t in tracks:
            genre = rows[t]["genre"]
            if counts.get(genre, 0) < args.per_genre:
                picked.append(t)
                counts[genre] = counts.get(genre, 0) + 1
        tracks = picked
    results, failures = {}, []
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for system in args.systems:
            started = perf_counter()
            predictor = load_predictor(ROOT, system)
            if predictor.model_id != campaign["models"][system]:
                failures.append({"system": system, "track_id": None, "reason": "model_id differs"})
            worst = 0.0
            with threadpool_limits(limits=1):
                for track in tracks:
                    path = ROOT / "data/raw" / rows[track]["audio_path"]
                    if sha256_file(path) != campaign["source_sha256"][track]:
                        raise ValueError(f"Audio changed: {track}")
                    error, reason = compare(stored(system, track), predictor.predict_file(path, mode="fma"))
                    if error is not None:
                        worst = max(worst, error)
                    if reason or (error is not None and error > TOLERANCE):
                        failures.append({"system": system, "track_id": track,
                                         "reason": reason or f"error {error:.3g}"})
            results[system] = {"tracks": len(tracks), "max_probability_error": worst,
                               "seconds": perf_counter() - started}
            print(f"{system}: {len(tracks)} tracks, max error {worst:.3g}, "
                  f"{results[system]['seconds']:.0f}s", flush=True)
    finally:
        torch.set_num_threads(previous)
    report = {"status": "passed" if not failures else "failed", "tolerance": TOLERANCE,
              "created_at": datetime.now(UTC).isoformat(), "split": "validation",
              "per_genre": args.per_genre, "systems": results, "failures": failures[:50],
              "failure_count": len(failures), "test_audio_loaded": False}
    write_json(ROOT / args.output, report)
    print(json.dumps({"status": report["status"], "failures": len(failures)}))
    return 0 if not failures else 1


if __name__ == "__main__":
    sys.exit(main())
