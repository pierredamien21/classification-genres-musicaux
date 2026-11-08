"""Check T4 report integrity, window means and independently recounted metrics."""

import json
from pathlib import Path

import numpy as np
from check_candidate_inference import read_csv

from music_genre.data.audit import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    pointer = json.loads((ROOT / "reports/latest_comparator_inference_check.json").read_text("utf-8"))
    assert sha256_file(ROOT / pointer["report"]) == pointer["report_sha256"]
    report = json.loads((ROOT / pointer["report"]).read_text("utf-8"))
    assert report["status"] == "complete" and not report["test_evaluated"] and not report["test_audio_loaded"]
    for path, digest in report["source_and_input_sha256"].items():
        assert sha256_file(ROOT / path) == digest, path
    prepared = json.loads((ROOT / "reports/PREPARATION_EVALUATION_FINALE.json").read_text("utf-8"))
    cohort = read_csv(ROOT / prepared["sample"]["path"])
    labels = prepared["labels"]
    counts = {}
    for name, system in report["systems"].items():
        pin = prepared["runs"][name]
        rows = read_csv(ROOT / pin["run"]["folder"] / "validation_predictions.csv")
        reference = {r["track_id"]: r for r in rows}
        confusion = np.zeros((8, 8), dtype=int)
        for row in rows:
            predicted = max(range(8), key=lambda i: float(row[f"p_{labels[i]}"]))
            confusion[labels.index(row["true_label"]), predicted] += 1
        denominators = confusion.sum(0) + confusion.sum(1)
        f1 = np.divide(2 * confusion.diagonal(), denominators, out=np.zeros(8), where=denominators != 0)
        metrics = system["full_validation"]
        assert len(rows) == metrics["tracks"] == 780
        np.testing.assert_array_equal(confusion, metrics["confusion_matrix"])
        assert abs(f1.mean() - metrics["macro_f1"]) < 1e-15
        assert confusion.trace() / len(rows) == metrics["accuracy"]
        assert [r["track_id"] for r in system["results"]] == [r["track_id"] for r in cohort]
        largest, windows = 0., 0
        for item, source in zip(system["results"], cohort, strict=True):
            assert item["source_sha256"] == source["source_sha256"]
            prediction = item["prediction"]
            assert prediction["labels"] == labels and prediction["model_id"] == system["model_id"]
            probabilities = np.array([w["scores"] for w in prediction["windows"]])
            assert probabilities.shape == (3, 8) and np.isfinite(probabilities).all() and (probabilities >= 0).all()
            np.testing.assert_allclose(probabilities.sum(1), 1, atol=1e-12, rtol=0)
            np.testing.assert_allclose(probabilities.mean(0), prediction["scores"], atol=1e-15, rtol=0)
            expected = np.array([float(reference[item["track_id"]][f"p_{label}"]) for label in labels])
            error = float(np.max(np.abs(np.array(prediction["scores"]) - expected)))
            assert error == item["track_probability_error"] and error <= 1e-6
            assert item["window_probability_error"] <= 1e-6
            largest, windows = max(largest, error), windows + len(probabilities)
        assert largest == system["max_track_probability_error"] and windows == system["panel_windows"] == 192
        counts[name] = {"tracks": 64, "windows": windows, "validation_tracks": len(rows),
                        "max_track_probability_error": largest}
    assert len(counts) == 6
    write_json(ROOT / "reports/COMPARATOR_INFERENCE_RECOUNT.json", {
        "status": "passed", "report": pointer, "verification_source_sha256": sha256_file(Path(__file__)),
        "systems": counts, "test_evaluated": False, "test_audio_loaded": False,
        "method": "source_hashes_window_means_saved_reference_and_manual_confusion_metrics"})
    print("Six systems verified: 1152 window probability vectors and six 780-track metric recounts")


if __name__ == "__main__":
    main()
