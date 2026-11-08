"""Independent verification of test predictions and optional final statistics.

No production inference, storage, aggregation or bootstrap implementation imported.
"""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data/interim/final-test-v1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(path.read_text("utf-8"))


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def manual_metrics(truth, predicted, labels):
    size = len(labels)
    require(len(truth) > 0 and len(truth) == len(predicted) and size > 1
            and len(set(labels)) == size, "Empty or misaligned metrics")
    confusion = [[0] * size for _ in labels]
    for actual, guess in zip(truth, predicted, strict=True):
        require(0 <= actual < size and 0 <= guess < size, "Invalid class index")
        confusion[actual][guess] += 1
    classes = {}
    for k, label in enumerate(labels):
        support, count = sum(confusion[k]), sum(row[k] for row in confusion)
        tp = confusion[k][k]
        classes[label] = {"precision": tp / count if count else 0.,
                          "recall": tp / support if support else 0.,
                          "f1-score": 2 * tp / (support + count) if support + count else 0.,
                          "support": support}
    return {"tracks": len(truth), "confusion_matrix": confusion, "per_class": classes,
            "macro_f1": math.fsum(v["f1-score"] for v in classes.values()) / size,
            "accuracy": sum(confusion[i][i] for i in range(size)) / len(truth)}


def recount_predictions():
    campaign, completion, progress = (read(FOLDER / name) for name in
                                       ("campaign.json", "completion.json", "progress.json"))
    require(campaign["split"] == "test" and completion["status"] == "inference_complete"
            and completion["test_evaluated"] is True and progress["status"] == "complete"
            and digest(FOLDER / "completion.json") == progress["completion_sha256"], "Incomplete test")
    identity = object_digest(campaign)
    require(identity == completion["campaign_sha256"], "Campaign identity differs")
    lock_ref = completion["lock"]
    require(digest(ROOT / lock_ref["report"]) == lock_ref["report_sha256"] == campaign["lock_sha256"],
            "Lock changed")
    locked = read(ROOT / lock_ref["report"])
    evidence = {}

    def pinned(name):
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), "Path escapes project")
        actual = digest(path)
        require(actual == locked["artifacts"][name], f"Locked evidence changed: {name}")
        evidence[name] = actual
        return path

    for name, expected in campaign["runner_sources"].items():
        require(digest(ROOT / name) == expected, "Runner changed")
        evidence[name] = expected
    validation = campaign["validation_evidence"]
    require(digest(ROOT / validation["path"]) == validation["sha256"], "Validation proof changed")
    validation_proof = read(ROOT / validation["path"])
    require(validation_proof["status"] == "validation_passed"
            and validation_proof["lock"]["report_sha256"] == campaign["lock_sha256"], "Invalid validation")
    launch = read(ROOT / "reports/FINAL_TEST_LAUNCH.json")
    require(launch["validation_completion_sha256"] == validation["sha256"], "Launch proof differs")
    independent = launch["independent_check"]
    require(digest(ROOT / independent["path"]) == independent["sha256"]
            and read(ROOT / independent["path"])["status"] == "passed", "Independent validation changed")
    manifest = csv_rows(pinned(locked["dataset"]["manifest"]))
    rows = sorted([r for r in manifest if r["split"] == "test"], key=lambda r: r["track_id"])
    ids = [r["track_id"] for r in rows]
    require(len(ids) == len(set(ids)) == 778 and ids == campaign["track_ids"]
            and ids == locked["dataset"]["track_ids"]["test"], "Test cohort differs")
    require(campaign["source_sha256"] == {r["track_id"]: r["source_sha256"] for r in rows},
            "Audio provenance differs")
    labels = locked["labels"]
    names = locked["protocol"]["linear_systems"] + locked["protocol"]["cnn_runs"]
    require(campaign["labels"] == labels and len(labels) == 8 and len(names) == 9
            and set(names) == set(campaign["models"]) == set(completion["systems"]), "Roster differs")
    identities = {}
    for stage in ("T4", "T5"):
        identities.update({n: v["model_id"] for n, v in read(pinned(locked["proofs"][stage]))["systems"].items()})
    require(campaign["models"] == identities, "Model identities differ")
    truth = [int(r["label_index"]) for r in rows]
    predictions, metrics, expected_paths = {}, {}, set()
    max_mean_error = 0.
    for name in names:
        require(completion["systems"][name]["tracks"] == 778, "Wrong completed count")
        guesses = []
        for row in rows:
            track = row["track_id"]
            relative = f"predictions/{name}/{track}.json"
            expected_paths.add(relative)
            path = FOLDER / relative
            require(digest(path) == completion["predictions"][relative], "Prediction file changed")
            record = read(path)
            payload = record["payload"]
            require(record["sha256"] == object_digest(payload) and payload["campaign"] == identity
                    and payload["system"] == name and payload["track_id"] == track
                    and payload["source_sha256"] == row["source_sha256"], "Record provenance differs")
            item = payload["prediction"]
            require(item["labels"] == labels and item["model_id"] == identities[name]
                    and item["mode"] == "fma" and item["schema_version"] == 1, "Prediction identity differs")
            windows = item["windows"]
            require(len(windows) == 3, "Wrong window count")
            for i, window in enumerate(windows):
                geometry = [window[k] * 16000 for k in ("start_seconds", "valid_seconds", "padded_seconds")]
                require(all(math.isfinite(x) and x >= 0 and abs(x - round(x)) <= 1e-7 for x in geometry),
                        "Invalid sample geometry")
                start, valid, padded = map(round, geometry)
                require(start == i * 160000 and valid + padded == 160000, "Unexpected geometry")
            for scores in [item["scores"]] + [w["scores"] for w in windows]:
                require(len(scores) == 8 and all(math.isfinite(x) and 0 <= x <= 1 for x in scores)
                        and abs(math.fsum(scores) - 1) <= 1e-6, "Invalid probability vector")
            mean = [math.fsum(w["scores"][i] for w in windows) / 3 for i in range(8)]
            error = max(abs(a - b) for a, b in zip(mean, item["scores"], strict=True))
            max_mean_error = max(max_mean_error, error)
            guess = max(range(8), key=lambda i: mean[i])
            require(error <= 1e-15 and guess == max(range(8), key=lambda i: item["scores"][i]),
                    "Window aggregation differs")
            guesses.append(guess)
        predictions[name] = guesses
        metrics[name] = manual_metrics(truth, guesses, labels)
    require(set(completion["predictions"]) == expected_paths == {
        p.relative_to(FOLDER).as_posix() for p in (FOLDER / "predictions").rglob("*.json")},
        "Unexpected or missing files")
    train = Counter(int(r["label_index"]) for r in manifest if r["split"] == "train")
    majority = min(range(8), key=lambda k: (-train[k], k))
    predictions["majority"] = [majority] * len(rows)
    metrics["majority"] = manual_metrics(truth, predictions["majority"], labels)
    result = {"status": "passed", "created_at": datetime.now(UTC).isoformat(),
              "completion_sha256": digest(FOLDER / "completion.json"), "lock_sha256": campaign["lock_sha256"],
              "predictions_verified": len(expected_paths), "window_vectors_verified": len(expected_paths) * 3,
              "max_window_mean_error": max_mean_error, "systems": metrics,
              "train_class_counts": [train[k] for k in range(8)], "majority_train_class": labels[majority],
              "evidence_sha256": evidence, "verification_source_sha256": digest(Path(__file__)),
              "method": "stdlib_fingerprints_fsum_window_means_manual_confusions_and_class_metrics",
              "audio_decoded": False, "test_evaluated": True}
    return result, predictions, rows, locked


def bounds(values):
    ordered = sorted(values)
    result = []
    for q in (.025, .975):
        at = (len(ordered) - 1) * q
        low, high = math.floor(at), math.ceil(at)
        result.append(ordered[low] + (at - low) * (ordered[high] - ordered[low]))
    return result


def verify_statistics(path, recount, predictions, rows, locked):
    import numpy as np
    from sklearn.metrics import accuracy_score, f1_score

    report = read(path)
    protocol = locked["protocol"]
    require(report["status"] == "complete" and report["split"] == "test" and report["test_evaluated"] is True
            and report["completion_sha256"] == recount["completion_sha256"], "Report identity differs")
    require(report["settings"] == protocol["bootstrap"] and report["candidate"] == protocol["candidate"]
            and report["comparisons"] == [protocol["primary_comparison"], *protocol["secondary_comparisons"]]
            and report["metric_groups"] == {"cnn20_mean": protocol["cnn_runs"]}, "Statistical protocol differs")
    require(report["train_class_counts"] == recount["train_class_counts"]
            and report["majority_train_class"] == recount["majority_train_class"]
            and report["labels"] == locked["labels"] and report["tracks"] == len(rows), "Truth differs")
    for name, expected in report["source_and_input_sha256"].items():
        require(digest(ROOT / name) == expected, f"Reporting source or evidence changed: {name}")
    for name, expected in report["exports_sha256"].items():
        require(digest(path.parent / name) == expected, "Report export changed")
    require(set(report["classification"]) == set(predictions), "Unexpected class metrics")
    points = {}
    for name, metrics in recount["systems"].items():
        actual = report["classification"][name]
        require(actual["confusion_matrix"] == metrics["confusion_matrix"] and actual["tracks"] == 778,
                "Confusion matrix differs")
        for key in ("macro_f1", "accuracy"):
            require(abs(actual[key] - metrics[key]) < 1e-12, "Global metric differs")
        for label in locked["labels"]:
            for key, value in metrics["per_class"][label].items():
                require(abs(actual["per_class"][label][key] - value) < 1e-12, "Class metric differs")
        points[name] = [metrics["macro_f1"], metrics["accuracy"]]
    points["cnn20_mean"] = [float(np.mean([points[n][k] for n in protocol["cnn_runs"]])) for k in range(2)]
    names = [*predictions, "cnn20_mean"]
    require(report["models"] == names and report["metrics"] == ["macro_f1", "accuracy"], "Model order differs")
    for name in names:
        for k, metric in enumerate(report["metrics"]):
            require(abs(points[name][k] - report["estimates"][name][metric]["point"]) < 1e-12, "Point differs")
    for k, metric in enumerate(report["metrics"]):
        require(abs(float(np.std([points[n][k] for n in protocol["cnn_runs"]], ddof=1))
                    - report["cnn_point_sample_std"][metric]) < 1e-12, "CNN sample dispersion differs")
    require(digest(ROOT / report["archive"]["path"]) == report["archive"]["sha256"], "Archive changed")
    with np.load(ROOT / report["archive"]["path"], allow_pickle=False) as archive:
        saved, counts, sizes = (archive[k] for k in ("replicate_scores", "artist_draw_counts", "replicate_track_counts"))
    artists = sorted({r["artist_id"] for r in rows})
    require(artists == report["artist_ids"] and report["artists"] == len(artists), "Artist roster differs")
    truth = np.array([int(r["label_index"]) for r in rows])
    groups = [np.array([i for i, r in enumerate(rows) if r["artist_id"] == a]) for a in artists]
    draws = np.random.Generator(np.random.PCG64(20261028)).choice(len(artists), size=(2000, len(artists)), replace=True)
    np.testing.assert_array_equal(counts, np.stack([(draws == a).sum(1) for a in range(len(artists))], axis=1))
    require(saved.shape == (2000, 11, 2), "Wrong bootstrap shape")
    reproduced = np.empty_like(saved)
    vectors = {n: np.array(v) for n, v in predictions.items()}
    missing = 0
    for b, draw in enumerate(draws):
        positions = np.concatenate([groups[a] for a in draw])
        require(len(positions) == sizes[b], "Resampled track count differs")
        missing += len(set(truth[positions])) < 8
        for m, name in enumerate(names):
            if name == "cnn20_mean":
                reproduced[b, m] = reproduced[b, [names.index(n) for n in protocol["cnn_runs"]]].mean(0)
            else:
                y, p = truth[positions], vectors[name][positions]
                reproduced[b, m] = [f1_score(y, p, labels=list(range(8)), average="macro", zero_division=0),
                                     accuracy_score(y, p)]
        if (b + 1) % 500 == 0:
            print(f"Verified {b + 1}/2000 expanded artist draws", flush=True)
    error = float(np.max(np.abs(reproduced - saved)))
    require(error < 1e-12 and missing == report["missing_class_replicates"], "Bootstrap scores differ")
    for m, name in enumerate(names):
        for k, metric in enumerate(report["metrics"]):
            expected = report["estimates"][name][metric]
            np.testing.assert_allclose(bounds(reproduced[:, m, k]), [expected["low"], expected["high"]], atol=1e-12, rtol=0)
    require(set(report["differences"]) == {f"{a} - {b}" for a, b in report["comparisons"]}, "Comparison roster differs")
    for left, right in report["comparisons"]:
        a, b = names.index(left), names.index(right)
        for k, metric in enumerate(report["metrics"]):
            expected = report["differences"][f"{left} - {right}"][metric]
            require(abs(expected["point"] - (points[left][k] - points[right][k])) < 1e-12, "Paired point differs")
            np.testing.assert_allclose(bounds(reproduced[:, a, k] - reproduced[:, b, k]),
                                       [expected["low"], expected["high"]], atol=1e-12, rtol=0)
    return {"status": "passed", "report": path.relative_to(ROOT).as_posix(), "report_sha256": digest(path),
            "created_at": datetime.now(UTC).isoformat(), "verification_source_sha256": digest(Path(__file__)),
            "predictions_verified": 7002, "replicates_verified": 2000, "metrics_verified": 44000,
            "max_metric_error": error, "missing_class_replicates": missing,
            "method": "manual_confusions_expanded_artist_tracks_sklearn_metrics_sorted_quantiles",
            "audio_decoded": False, "test_evaluated": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", help="Optional completed report.json, relative to project root")
    args = parser.parse_args()
    recount, predictions, rows, locked = recount_predictions()
    (ROOT / "reports/FINAL_TEST_RECOUNT.json").write_text(json.dumps(recount, indent=2) + "\n", encoding="utf-8")
    print("PASS: 7002 files, 21006 window vectors, nine models and train majority recounted", flush=True)
    if args.report:
        path = (ROOT / args.report).resolve()
        require(path.is_relative_to(ROOT / "reports"), "Report outside reports directory")
        check = verify_statistics(path, recount, predictions, rows, locked)
        (ROOT / "reports/FINAL_TEST_REPORT_CHECK.json").write_text(json.dumps(check, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(check), flush=True)


if __name__ == "__main__":
    main()
