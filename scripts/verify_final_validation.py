"""Independent T7 dry-run recount: no inference/store/metric implementation imports."""

import csv
import hashlib
import json
import math
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / "data/interim/final-validation-v1"


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


def require(condition, message):
    if not condition:
        raise ValueError(message)


def recount_prediction(item, labels, model_id):
    require(item["schema_version"] == 1 and item["mode"] == "fma"
            and item["labels"] == labels and item["model_id"] == model_id, "Identity differs")
    windows = item["windows"]
    require(len(windows) == 3, "Expected three windows")
    for i, window in enumerate(windows):
        samples = []
        for key in ("start_seconds", "valid_seconds", "padded_seconds"):
            value = window[key] * 16000
            require(math.isfinite(value) and value >= 0 and abs(value - round(value)) < 1e-7,
                    "Invalid sample geometry")
            samples.append(round(value))
        require(samples[0] == i * 160000 and samples[1] + samples[2] == 160000,
                "Unexpected geometry")
    for scores in [item["scores"]] + [w["scores"] for w in windows]:
        require(len(scores) == len(labels)
                and all(math.isfinite(p) and 0 <= p <= 1 for p in scores)
                and abs(math.fsum(scores) - 1) <= 1e-6, "Invalid probabilities")
    mean = [math.fsum(w["scores"][i] for w in windows) / 3 for i in range(len(labels))]
    require(max(abs(a - b) for a, b in zip(mean, item["scores"], strict=True)) <= 1e-15,
            "Window mean differs")
    return mean


def main():
    campaign = read(FOLDER / "campaign.json")
    completion = read(FOLDER / "completion.json")
    progress = read(FOLDER / "progress.json")
    require(progress["status"] == "complete"
            and digest(FOLDER / "completion.json") == progress["completion_sha256"]
            and completion["status"] == "validation_passed"
            and completion["test_evaluated"] is False, "Incomplete or changed dry run")
    identity = object_digest(campaign)
    require(identity == completion["campaign_sha256"] and campaign["split"] == "validation",
            "Campaign identity differs")
    lock_ref = read(ROOT / "reports/latest_final_evaluation_lock.json")
    require(lock_ref["report_sha256"] == campaign["lock_sha256"]
            and digest(ROOT / lock_ref["report"]) == campaign["lock_sha256"], "Lock changed")
    locked = read(ROOT / lock_ref["report"])
    evidence = {}

    def pinned(name):
        path = (ROOT / name).resolve()
        require(path.is_relative_to(ROOT), "Path escapes root")
        actual = digest(path)
        require(actual == locked["artifacts"][name], f"Locked evidence changed: {name}")
        evidence[name] = actual
        return path

    for name, expected in campaign["runner_sources"].items():
        require(digest(ROOT / name) == expected, f"Runner changed: {name}")
        evidence[name] = expected
    rows = csv_rows(pinned(locked["dataset"]["manifest"]))
    validation = {r["track_id"]: r for r in rows if r["split"] == "validation"}
    ids = sorted(validation)
    require(len(ids) == 780 and ids == campaign["track_ids"]
            and ids == locked["dataset"]["track_ids"]["validation"], "Wrong validation cohort")
    require(campaign["source_sha256"] == {t: validation[t]["source_sha256"] for t in ids},
            "Audio provenance differs")
    labels = locked["labels"]
    names = locked["protocol"]["linear_systems"] + locked["protocol"]["cnn_runs"]
    require(campaign["labels"] == labels and len(labels) == 8
            and set(campaign["models"]) == set(names) and len(names) == 9,
            "Model roster or labels differ")
    historical = {}
    for stage in ("T4", "T5"):
        historical.update(read(pinned(locked["proofs"][stage]))["systems"])
    expected_paths, results = set(), {}
    for name in names:
        pin = locked["models"][name]
        reference_rows = csv_rows(pinned(pin["run"]["folder"] + "/validation_predictions.csv"))
        reference = {r["track_id"]: r for r in reference_rows}
        require(len(reference_rows) == len(reference) == 780 and sorted(reference) == ids,
                "Reference cohort differs")
        model_id = historical[name]["model_id"]
        require(campaign["models"][name] == model_id, "Model identity differs")
        confusion = [[0] * 8 for _ in labels]
        largest, largest_mean, disagreements = 0., 0., 0
        for track in ids:
            relative = f"predictions/{name}/{track}.json"
            expected_paths.add(relative)
            path = FOLDER / relative
            require(digest(path) == completion["predictions"][relative], "Prediction file changed")
            record = read(path)
            payload = record["payload"]
            require(record["sha256"] == object_digest(payload)
                    and payload["campaign"] == identity and payload["system"] == name
                    and payload["track_id"] == track
                    and payload["source_sha256"] == validation[track]["source_sha256"],
                    "Record provenance differs")
            item = payload["prediction"]
            mean = recount_prediction(item, labels, model_id)
            expected = [float(reference[track][f"p_{label}"]) for label in labels]
            require(all(math.isfinite(p) and 0 <= p <= 1 for p in expected)
                    and abs(math.fsum(expected) - 1) <= 1e-6, "Invalid reference probabilities")
            error = max(abs(a - b) for a, b in zip(item["scores"], expected, strict=True))
            largest = max(largest, error)
            largest_mean = max(largest_mean, max(abs(a - b) for a, b in zip(mean, expected, strict=True)))
            predicted = max(range(8), key=lambda i: item["scores"][i])
            expected_class = max(range(8), key=lambda i: expected[i])
            require(predicted == max(range(8), key=lambda i: mean[i]), "Recounted class differs")
            disagreements += int(predicted != expected_class)
            truth = int(validation[track]["label_index"])
            require(labels[truth] == reference[track]["true_label"], "Reference truth differs")
            confusion[truth][predicted] += 1
        require(largest <= 1e-6 and largest_mean <= 1e-6 and disagreements == 0,
                "Independent reference comparison failed")
        claimed = completion["systems"][name]
        require(claimed == {"compared_tracks": 780, "max_probability_error": largest,
                            "class_disagreements": 0}, "Reported comparison differs")
        f1 = []
        for k in range(8):
            denominator = sum(confusion[k]) + sum(row[k] for row in confusion)
            f1.append(2 * confusion[k][k] / denominator if denominator else 0.)
        metrics = {"macro_f1": math.fsum(f1) / 8,
                   "accuracy": sum(confusion[k][k] for k in range(8)) / 780}
        old_metrics = historical[name]["full_validation"]
        require(confusion == old_metrics["confusion_matrix"] and all(
            abs(value - old_metrics[key]) <= 1e-15 for key, value in metrics.items()),
            "Independently recounted metrics differ")
        results[name] = {"tracks": 780, "windows": 2340, "max_probability_error": largest,
                         "max_fsum_probability_error": largest_mean, "class_disagreements": 0,
                         "confusion_matrix": confusion, **metrics}
    require(set(completion["predictions"]) == expected_paths and expected_paths == {
        p.relative_to(FOLDER).as_posix() for p in (FOLDER / "predictions").rglob("*.json")},
        "Unexpected or missing prediction files")
    report = {"status": "passed", "created_at": datetime.now(UTC).isoformat(),
              "campaign_sha256": identity, "completion_sha256": digest(FOLDER / "completion.json"),
              "lock_sha256": campaign["lock_sha256"], "predictions_verified": len(expected_paths),
              "window_vectors_verified": 21060, "systems": results, "evidence_sha256": evidence,
              "verification_source_sha256": digest(Path(__file__)),
              "method": "stdlib_file_hashes_fsum_window_means_manual_confusions_fixed_labels",
              "audio_decoded": False, "test_evaluated": False,
              "scope": "Prediction files and supporting evidence; full dataset integrity checked again by test runner"}
    (ROOT / "reports/FINAL_VALIDATION_RECOUNT.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("PASS: 7020 predictions, 21060 windows, nine reference comparisons and manual metric recounts")


if __name__ == "__main__":
    main()
