"""Planning only: detectable macro-F1 differences for the V2 test design (64 tracks x 8 genres,
one track per artist), simulated from the already consumed V1 test predictions.

Each replicate draws 64 V1 test tracks per genre with replacement (tracks treated as
independent, as one-track-per-artist sampling intends) and scores paired systems.
Writes reports/v2/power.json. Never used to choose a model.
"""

import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from scipy.stats import norm

from music_genre.data.audit import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "data/interim/final-test-v1"
PAIRS = (("vit_block6_hpss", "vit"), ("vit_block6_hpss", "vit_block6"), ("vit_block6_hpss", "cnn20_seed42"))
PER_GENRE, REPLICATES, SEED, POWER, HYPOTHESES = 64, 2000, 20261108, 0.8, 3


def macro_f1(truth, predicted, classes):
    scores = []
    for k in range(classes):
        tp = np.sum((truth == k) & (predicted == k))
        denominator = np.sum(truth == k) + np.sum(predicted == k)
        scores.append(2 * tp / denominator if denominator else 0.0)
    return float(np.mean(scores))


def main():
    campaign = json.loads((CAMPAIGN / "campaign.json").read_text("utf-8"))
    baseline = json.loads((ROOT / "data/manifests/latest_baseline.json").read_text())
    with (ROOT / baseline["folder"] / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        truth_of = {r["track_id"]: int(r["label_index"]) for r in csv.DictReader(stream)}
    tracks = sorted(campaign["track_ids"])
    truth = np.array([truth_of[t] for t in tracks])
    systems = sorted({s for pair in PAIRS for s in pair})
    predicted = {s: np.array([int(np.argmax(json.loads((CAMPAIGN / "predictions" / s / f"{t}.json")
                                              .read_text("utf-8"))["payload"]["prediction"]["scores"]))
                              for t in tracks]) for s in systems}
    classes = len(campaign["labels"])
    by_class = [np.flatnonzero(truth == k) for k in range(classes)]
    rng = np.random.Generator(np.random.PCG64(SEED))
    draws = [np.concatenate([rng.choice(rows, PER_GENRE) for rows in by_class]) for _ in range(REPLICATES)]
    alpha = 0.05 / HYPOTHESES  # Holm's strictest step
    multiplier = norm.ppf(1 - alpha / 2) + norm.ppf(POWER)
    results = {}
    for left, right in PAIRS:
        differences = np.array([macro_f1(truth[d], predicted[left][d], classes)
                                - macro_f1(truth[d], predicted[right][d], classes) for d in draws])
        sd = float(differences.std(ddof=1))
        results[f"{left} - {right}"] = {"v1_test_difference": macro_f1(truth, predicted[left], classes)
                                        - macro_f1(truth, predicted[right], classes),
                                        "simulated_sd": sd, "minimum_detectable_difference": multiplier * sd}
    write_json(ROOT / "reports/v2/power.json", {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(), "design": {
            "per_genre": PER_GENRE, "genres": classes, "one_track_per_artist": True},
        "replicates": REPLICATES, "seed": SEED, "power": POWER, "alpha_two_sided_holm_strictest": alpha,
        "comparisons": results, "source": "V1 test predictions (already consumed); planning only",
        "script_sha256": sha256_file(Path(__file__))})
    print(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
