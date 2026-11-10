"""V2 phase 4: temperature calibration and "unknown genre" threshold from out-of-fold predictions.

  python scripts/calibrate_v2.py --run runs/v2-cv/<system>-<timestamp> [--robustness runs/v2-robustness/<...>]

Honest estimates are cross-fitted: each fold is calibrated, and its rejection threshold set, from
the other folds only. The final temperature and threshold (for the V2 test) are then fitted on all
out-of-fold tracks. With --robustness, the final pair is applied to every degraded condition to
show how many in-domain degraded tracks would be rejected. Development predictions only.
"""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.calibration import (
    cross_fit_temperature,
    expected_calibration_error,
    fit_temperature,
    rejection_threshold,
    temperature_scale,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/calibration_v2.json"


def read_oof(path, labels):
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    probs = np.array([[float(r[f"p_{label}"]) for label in labels] for r in rows])
    return ([r["track_id"] for r in rows], np.array([int(r["label_index"]) for r in rows]),
            np.array([int(r["fold"]) for r in rows]), probs)


def quality(probs, truth, bins):
    picked = np.clip(probs[np.arange(len(truth)), truth], 1e-12, 1)
    return {"ece": expected_calibration_error(probs, truth, bins), "nll": float(-np.mean(np.log(picked))),
            "brier": float(np.mean(np.sum((probs - np.eye(probs.shape[1])[truth]) ** 2, axis=1))),
            "mean_top_probability": float(probs.max(axis=1).mean()),
            "accuracy": float(np.mean(probs.argmax(axis=1) == truth))}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--robustness")
    args = parser.parse_args()
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    labels = json.loads((ROOT / "configs/labels.json").read_text())["labels"]
    run = (ROOT / args.run).resolve()
    result = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if sha256_file(run / "oof_tracks.csv") != result["oof_tracks_sha256"]:
        raise ValueError("Out-of-fold file changed")
    ids, truth, folds, probs = read_oof(run / "oof_tracks.csv", labels)
    rate, bins = config["rejection_rate_in_domain"], config["ece_bins"]
    calibrated, temperatures = cross_fit_temperature(probs, truth, folds)
    per_fold = []
    for k in sorted(temperatures):
        held = folds == k
        threshold = rejection_threshold(calibrated[~held].max(axis=1), rate)
        accepted = held & (calibrated.max(axis=1) >= threshold)
        per_fold.append({
            "fold": k, "temperature": temperatures[k], "threshold": threshold,
            "rejected_share": float(1 - accepted.sum() / held.sum()),
            "macro_f1_all": float(f1_score(truth[held], probs[held].argmax(axis=1), average="macro")),
            "macro_f1_accepted": float(f1_score(truth[accepted], probs[accepted].argmax(axis=1), average="macro")),
            "accuracy_rejected": float(np.mean(probs[held & ~accepted].argmax(axis=1) == truth[held & ~accepted])),
        })
    temperature = fit_temperature(probs, truth, tuple(config["temperature_bounds"]))
    final = temperature_scale(probs, temperature)
    threshold = rejection_threshold(final.max(axis=1), rate)
    degraded = {}
    if args.robustness:
        folder = (ROOT / args.robustness).resolve()
        robustness = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        if robustness["run"] != run.relative_to(ROOT).as_posix():
            raise ValueError("Robustness report belongs to another run")
        for name, info in robustness["conditions"].items():
            if sha256_file(folder / f"oof_{name}.csv") != info["oof_sha256"]:
                raise ValueError(f"Robustness file changed: {name}")
            _, condition_truth, _, condition_probs = read_oof(folder / f"oof_{name}.csv", labels)
            scaled = temperature_scale(condition_probs, temperature)
            degraded[name] = {"rejected_share": float(np.mean(scaled.max(axis=1) < threshold)),
                              **quality(scaled, condition_truth, bins)}
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(), "system": result["system"],
        "run": run.relative_to(ROOT).as_posix(), "run_report_sha256": sha256_file(run / "report.json"),
        "config": config, "config_sha256": sha256_file(ROOT / CONFIG),
        "source_sha256": {p: sha256_file(ROOT / p) for p in (
            "scripts/calibrate_v2.py", "src/music_genre/evaluation/calibration.py")},
        "tracks": len(ids), "uncalibrated": quality(probs, truth, bins),
        "cross_fitted": quality(calibrated, truth, bins), "folds": per_fold,
        "cross_fitted_rejected_share": float(np.mean([f["rejected_share"] for f in per_fold])),
        "final": {"temperature": temperature, "threshold_top_probability": threshold,
                  "fitted_on": "all out-of-fold development tracks"},
        "degraded_conditions": degraded, "v2_test_audio_loaded": False, "out_of_domain_loaded": False,
    }
    out = ROOT / "runs/v2-calibration" / f"{result['system']}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    out.mkdir(parents=True)
    write_json(out / "report.json", report)
    print(json.dumps({k: report[k] for k in ("uncalibrated", "cross_fitted", "cross_fitted_rejected_share", "final")},
                     indent=1))


if __name__ == "__main__":
    main()
