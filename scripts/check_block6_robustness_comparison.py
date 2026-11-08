"""Independent recount from stored predictions and regeneration of input PCM."""

import hashlib
import json
from pathlib import Path

import numpy as np
from check_candidate_inference import read_csv

from music_genre.audio.processing import load_audio
from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.robustness import perturb_audio

ROOT = Path(__file__).resolve().parents[1]


def main():
    pointer = json.loads((ROOT / "reports/latest_block6_robustness_comparison.json").read_text("utf-8"))
    path = ROOT / pointer["report"]
    assert sha256_file(path) == pointer["report_sha256"]
    report = json.loads(path.read_text("utf-8"))
    assert report["status"] == "complete" and not report["test_evaluated"] and not report["test_audio_loaded"]
    for name, digest in report["source_and_input_sha256"].items():
        assert sha256_file(ROOT / name) == digest, name
    hpss = json.loads((ROOT / "reports/ROBUSTNESS_PANEL.json").read_text("utf-8"))
    labels = report["labels"]
    counts = {}
    for name, collection in (("original", report["results"]), ("hpss", hpss["results"])):
        assert [r["track_id"] for r in collection] == [r["track_id"] for r in report["results"]]
        assert len(collection) == len({r["track_id"] for r in collection}) == 64
        counts[name] = {}
        for condition in report["conditions"]:
            confusion = np.zeros((8, 8), dtype=int)
            flips = corrected = regressed = windows = 0
            for track in collection:
                prediction = track["conditions"][condition]["prediction"]
                scores = np.array([w["scores"] for w in prediction["windows"]])
                assert scores.shape == (3, 8) and np.isfinite(scores).all()
                assert np.all(scores >= 0)
                np.testing.assert_allclose(scores.sum(1), 1, rtol=0, atol=1e-12)
                np.testing.assert_allclose(scores.mean(0), prediction["scores"], rtol=0, atol=1e-15)
                y = labels.index(track["genre"])
                clean = int(np.argmax(track["conditions"]["identity"]["prediction"]["scores"]))
                changed = int(np.argmax(prediction["scores"]))
                confusion[y, changed] += 1
                flips += clean != changed
                corrected += clean != y and changed == y
                regressed += clean == y and changed != y
                windows += len(scores)
            denominator = confusion.sum(0) + confusion.sum(1)
            f1 = np.divide(2 * confusion.diagonal(), denominator, out=np.zeros(8), where=denominator != 0)
            expected = report["summary"][name][condition]
            assert abs(f1.mean() - expected["metrics"]["macro_f1"]) < 1e-15
            assert confusion.trace() / 64 == expected["metrics"]["accuracy"]
            assert flips == expected["flipped_tracks"]
            assert corrected == expected["paired_outcomes"]["corrected"]
            assert regressed == expected["paired_outcomes"]["regressed"]
            for i, label in enumerate(labels):
                assert abs(f1[i] - expected["metrics"]["per_class"][label]["f1-score"]) < 1e-15
            counts[name][condition] = {"windows": windows, "macro_f1": float(f1.mean()),
                                       "accuracy": float(confusion.trace() / 64), "flips": flips}
        for condition in report["conditions"]:
            drop = counts[name]["identity"]["macro_f1"] - counts[name][condition]["macro_f1"]
            assert abs(drop - report["summary"][name][condition]["macro_f1_drop"]) < 1e-15
    prepared = json.loads((ROOT / "reports/PREPARATION_EVALUATION_FINALE.json").read_text("utf-8"))
    cohort = read_csv(ROOT / prepared["sample"]["path"])
    assert [r["track_id"] for r in cohort] == [r["track_id"] for r in report["results"]]
    snrs = []
    for row, track in zip(cohort, report["results"], strict=True):
        assert row["split"] == "validation" and row["genre"] == track["genre"]
        audio_path = ROOT / "data/raw" / row["audio_path"]
        assert sha256_file(audio_path) == row["source_sha256"] == track["source_sha256"]
        original = load_audio(audio_path, 16000).samples
        for condition in report["conditions"]:
            samples = perturb_audio(original, condition, row["track_id"], seed=20261028)
            assert hashlib.sha256(samples.tobytes()).hexdigest() == track["conditions"][condition]["audio"]["pcm_sha256"]
            if condition == "white_noise_snr_20_db":
                snr = 10 * np.log10(np.mean(original.astype(float) ** 2) / np.mean((samples.astype(float) - original) ** 2))
                assert abs(snr - 20) < 1e-4
                snrs.append(float(snr))
    write_json(ROOT / "reports/BLOCK6_ROBUSTNESS_CHECK.json", {
        "status": "passed", "comparison": pointer, "verification_source_sha256": sha256_file(Path(__file__)),
        "source_hashes_verified": len(report["source_and_input_sha256"]), "recount": counts,
        "pcm_hashes_reproduced": 192, "snr_min_db": min(snrs), "snr_max_db": max(snrs),
        "test_evaluated": False, "test_audio_loaded": False,
    })
    print("Independent metrics/window recount, source hashes and 192 PCM hashes: passed")


if __name__ == "__main__":
    main()
