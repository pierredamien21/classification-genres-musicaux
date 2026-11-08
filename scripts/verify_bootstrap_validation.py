"""Recompute all bootstrap draws by explicit track expansion and sklearn metrics."""

import json
from pathlib import Path
from time import perf_counter

import numpy as np
from check_candidate_inference import read_csv
from sklearn.metrics import accuracy_score, f1_score

from music_genre.data.audit import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def bounds(values):
    ordered = sorted(values)
    result = []
    for q in (.025, .975):
        position = (len(ordered) - 1) * q
        low, high = int(np.floor(position)), int(np.ceil(position))
        result.append(ordered[low] + (position - low) * (ordered[high] - ordered[low]))
    return result


def main():
    start = perf_counter()
    pointer = json.loads((ROOT / "reports/latest_bootstrap_validation.json").read_text("utf-8"))
    assert sha256_file(ROOT / pointer["report"]) == pointer["report_sha256"]
    report = json.loads((ROOT / pointer["report"]).read_text("utf-8"))
    for path, digest in report["source_and_input_sha256"].items():
        assert sha256_file(ROOT / path) == digest, path
    assert not report["audio_loaded"] and not report["test_evaluated"]
    assert sha256_file(ROOT / report["archive"]["path"]) == report["archive"]["sha256"]
    with np.load(ROOT / report["archive"]["path"], allow_pickle=False) as data:
        scores, counts, sizes = (data[k] for k in ("replicate_scores", "artist_draw_counts", "replicate_track_counts"))
    prepared = json.loads((ROOT / "reports/PREPARATION_EVALUATION_FINALE.json").read_text("utf-8"))
    manifest = read_csv(ROOT / prepared["baseline"]["folder"] / "manifest.csv")
    tracks = sorted([r for r in manifest if r["split"] == "validation"], key=lambda r: r["track_id"])
    truth = np.array([int(r["label_index"]) for r in tracks])
    groups = [np.array([i for i, row in enumerate(tracks) if row["artist_id"] == artist]) for artist in report["artist_ids"]]
    draws = np.random.Generator(np.random.PCG64(20261028)).choice(
        np.arange(len(groups)), size=(2000, len(groups)), replace=True)
    assert counts.shape == (2000, len(groups))
    np.testing.assert_array_equal(counts, np.stack([(draws == a).sum(1) for a in range(len(groups))], axis=1))
    predictions = {}
    for name in report["models"]:
        if name == "cnn20_mean":
            continue
        if name == "majority":
            # Derive from train independently of the saved majority label.
            from collections import Counter
            train = Counter(int(r["label_index"]) for r in manifest if r["split"] == "train")
            label = min(train, key=lambda i: (-train[i], i))
            predictions[name] = np.full(len(tracks), label)
        else:
            rows = sorted(read_csv(ROOT / prepared["runs"][name]["run"]["folder"] / "validation_predictions.csv"),
                          key=lambda row: row["track_id"])
            assert [r["track_id"] for r in rows] == [r["track_id"] for r in tracks]
            predictions[name] = np.array([max(range(8), key=lambda i: float(r[f"p_{report['labels'][i]}"])) for r in rows])
    reproduced = np.empty_like(scores)
    missing = 0
    for b, draw in enumerate(draws):
        positions = np.concatenate([groups[a] for a in draw])
        assert len(positions) == sizes[b]
        missing += len(set(truth[positions])) < 8
        for m, name in enumerate(report["models"]):
            if name == "cnn20_mean":
                indices = [report["models"].index(k) for k in report["metric_groups"][name]]
                reproduced[b, m] = reproduced[b, indices].mean(0)
            else:
                y, p = truth[positions], predictions[name][positions]
                reproduced[b, m] = [f1_score(y, p, labels=list(range(8)), average="macro", zero_division=0), accuracy_score(y, p)]
        if (b + 1) % 500 == 0:
            print(f"Verified {b + 1}/2000 expanded artist draws", flush=True)
    error = float(np.max(np.abs(reproduced - scores)))
    assert error < 1e-12 and missing == report["missing_class_replicates"]
    for m, name in enumerate(report["models"]):
        for k, metric in enumerate(report["metrics"]):
            expected = report["estimates"][name][metric]
            np.testing.assert_allclose(bounds(reproduced[:, m, k]), [expected["low"], expected["high"]], rtol=0, atol=1e-12)
    for left, right in report["comparisons"]:
        a, b = (report["models"].index(name) for name in (left, right))
        for k, metric in enumerate(report["metrics"]):
            expected = report["differences"][f"{left} - {right}"][metric]
            np.testing.assert_allclose(bounds(reproduced[:, a, k] - reproduced[:, b, k]),
                                       [expected["low"], expected["high"]], rtol=0, atol=1e-12)
    write_json(ROOT / "reports/BOOTSTRAP_VALIDATION_CHECK.json", {
        "status": "passed", "report": pointer, "verification_source_sha256": sha256_file(Path(__file__)),
        "replicates_verified": 2000, "metrics_verified": 44000, "max_metric_error": error,
        "missing_class_replicates": missing, "quantiles_recomputed_by_sorted_interpolation": True,
        "method": "explicit_tracks_expanded_from_PCG64_artist_draws_sklearn_fixed_eight_labels",
        "elapsed_seconds": perf_counter() - start, "audio_loaded": False, "test_evaluated": False})
    print(f"All scores and paired intervals verified; maximum error {error:.3g}", flush=True)


if __name__ == "__main__":
    main()
