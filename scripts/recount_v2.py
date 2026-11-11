"""V2 phase 5: independent recount of an evaluate_v2.py report from its predictions file.

  python scripts/recount_v2.py --evaluation runs/v2-final-evaluation

Shares no statistics code with evaluate_v2.py: scikit-learn metrics, its own temperature scaling and
ECE, scipy's paired BCa bootstrap (its own random stream, so intervals agree only up to Monte Carlo
error). Point estimates must agree to 1e-9; interval bounds within the configured tolerance, and
every "interval excludes 0" decision must agree. Writes recount.json next to the report.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from music_genre.data.audit import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
POINT_TOLERANCE, BOUND_TOLERANCE = 1e-9, 0.01


def ece(probs, truth, bins):
    confidence = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == truth).astype(float)
    total = 0.0
    for b in range(bins):
        low, high = b / bins, (b + 1) / bins
        members = (confidence > low) & (confidence <= high) if b else (confidence <= high)
        if members.any():
            total += members.sum() / len(truth) * abs(confidence[members].mean() - correct[members].mean())
    return total


def scaled(probs, temperature):
    z = np.log(np.clip(probs, 1e-12, 1)) / temperature
    z = np.exp(z - z.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluation", required=True)
    args = parser.parse_args()
    folder = (ROOT / args.evaluation).resolve()
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if sha256_file(folder / "predictions.csv") != report["predictions_sha256"]:
        raise ValueError("Predictions file changed")
    config = json.loads((ROOT / "configs/evaluation_v2.json").read_text(encoding="utf-8"))
    labels = json.loads((ROOT / "configs/labels.json").read_text())["labels"]
    calibration = json.loads((ROOT / config["calibration"]["report"]).read_text(encoding="utf-8"))["final"]
    with (folder / "predictions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    systems = list(dict.fromkeys(r["system"] for r in rows))
    tracks = {}
    for r in rows:
        tracks.setdefault(r["track_id"], {"role": r["role"], "genre": r["genre"]})[r["system"]] = np.array(
            [float(r[f"p_{label}"]) for label in labels])
    if report["mode"] == "final":
        with (ROOT / config["cohorts"]).open(encoding="utf-8", newline="") as stream:
            cohort = {r["track_id"]: (r["role"], r["genre"]) for r in csv.DictReader(stream)}
        if cohort != {t: (v["role"], v["genre"]) for t, v in tracks.items()}:
            raise ValueError("Predictions do not cover exactly the locked cohorts")
    test = [t for t, v in tracks.items() if v["role"] == "test"]
    truth = np.array([labels.index(tracks[t]["genre"]) for t in test])
    probs = {s: np.array([tracks[t][s] for t in test]) for s in systems}
    predicted = {s: p.argmax(axis=1) for s, p in probs.items()}
    k = range(len(labels))

    def f1(t, p):
        return f1_score(t, p, labels=k, average="macro", zero_division=0)

    differences, decisions, warnings = {}, {}, []

    def compare(name, mine, theirs, tolerance=POINT_TOLERANCE):
        differences[name] = float(abs(np.asarray(mine, dtype=float) - np.asarray(theirs, dtype=float)).max())
        if differences[name] > tolerance and tolerance == POINT_TOLERANCE:
            raise ValueError(f"Recount disagrees on {name}: {differences[name]}")
        if differences[name] > tolerance:  # interval bounds: Monte Carlo error, reported not fatal
            warnings.append(f"{name} differs by {differences[name]:.4f}")

    for s in systems:
        theirs = report["systems"][s]
        compare(f"{s}/macro_f1", f1(truth, predicted[s]), theirs["macro_f1"]["point"])
        compare(f"{s}/accuracy", accuracy_score(truth, predicted[s]), theirs["accuracy"])
        compare(f"{s}/per_genre_f1", f1_score(truth, predicted[s], labels=k, average=None, zero_division=0),
                [theirs["per_genre_f1"][label] for label in labels])
        compare(f"{s}/confusion", confusion_matrix(truth, predicted[s], labels=k), theirs["confusion_matrix"])
    boot = config["bootstrap"]
    rng = np.random.default_rng(boot["seed"])
    for name, h in config["hypotheses"].items():
        others = h["minus_mean_of"]

        def statistic(t, a, *bs):
            return f1(t, a) - np.mean([f1(t, b) for b in bs])

        data = (truth, predicted[h["system"]], *(predicted[b] for b in others))
        theirs = report["hypotheses"][name]
        compare(f"{name}/point", statistic(*data), theirs["point"])
        interval = stats.bootstrap(data, statistic, paired=True, vectorized=False, n_resamples=boot["replicates"],
                                   confidence_level=boot["level"], method="BCa", rng=rng).confidence_interval
        compare(f"{name}/bounds", [interval.low, interval.high], [theirs["low"], theirs["high"]], BOUND_TOLERANCE)
        decisions[name] = {"recount_excludes_zero": bool(interval.low > 0 or interval.high < 0),
                           "report_excludes_zero": bool(theirs["low"] > 0 or theirs["high"] < 0),
                           "recount_interval": [float(interval.low), float(interval.high)]}
        if decisions[name]["recount_excludes_zero"] != decisions[name]["report_excludes_zero"]:
            warnings.append(f"{name}: the recount interval reaches another decision (bound near 0)")
    candidate = probs[config["candidate"]]
    one_hot = np.eye(len(labels))[truth]
    for kind, p in (("uncalibrated", candidate), ("calibrated", scaled(candidate, calibration["temperature"]))):
        theirs = report["calibration_on_test"][kind]
        compare(f"calibration/{kind}/ece", ece(p, truth, config["calibration"]["ece_bins"]), theirs["ece"])
        compare(f"calibration/{kind}/brier", np.mean(np.sum((p - one_hot) ** 2, axis=1)), theirs["brier"])
    confidence = {t: scaled(tracks[t][config["candidate"]][None], calibration["temperature"]).max()
                  for t in tracks}
    inside = np.array([confidence[t] for t in test])
    for role, key in (("ood", "ood"), ("gospel_probe", "gospel_probe")):
        outside = np.array([confidence[t] for t, v in tracks.items() if v["role"] == role])
        theirs = report["out_of_domain"][key]
        compare(f"{key}/auroc", roc_auc_score(np.r_[np.ones(len(inside)), np.zeros(len(outside))],
                                              np.r_[inside, outside]), theirs["auroc"])
        compare(f"{key}/fpr", np.mean(outside >= calibration["threshold_top_probability"]), theirs["fpr_at_threshold"])
    excluded = set()
    if report["mode"] == "final":
        with (ROOT / config["decoder_sensitivity"]["csv"]).open(encoding="utf-8", newline="") as stream:
            excluded = {r["track_id"] for r in csv.DictReader(stream) if r["in_final_cohorts"] == "True"}
    kept = np.array([t not in excluded for t in test])
    for s in systems:
        compare(f"sensitivity/{s}", f1(truth[kept], predicted[s][kept]), report["decoder_sensitivity"]["macro_f1"][s])
    result = {"schema_version": 1, "evaluation": folder.relative_to(ROOT).as_posix(),
              "report_sha256": sha256_file(folder / "report.json"), "mode": report["mode"],
              "source_sha256": sha256_file(Path(__file__)), "status": "agrees" if not warnings else "see warnings",
              "warnings": warnings,
              "point_tolerance": POINT_TOLERANCE, "bound_tolerance": BOUND_TOLERANCE,
              "max_point_difference": max(v for n, v in differences.items() if not n.endswith("bounds")),
              "max_bound_difference": max(v for n, v in differences.items() if n.endswith("bounds")),
              "decisions": decisions, "test_tracks": len(test)}
    write_json(folder / "recount.json", result)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
