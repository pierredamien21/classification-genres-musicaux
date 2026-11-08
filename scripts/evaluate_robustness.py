"""Run the predeclared validation panel through unchanged public audio inference."""

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
from music_genre.inference.predict import CandidatePredictor

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = ROOT / "reports/ROBUSTNESS_PROGRESS.json"


def main():
    started = perf_counter()
    prepared_path = ROOT / "reports/PREPARATION_EVALUATION_FINALE.json"
    prepared = json.loads(prepared_path.read_text("utf-8"))
    protocol_path = ROOT / prepared["protocol_path"]
    if sha256_file(protocol_path) != prepared["protocol_sha256"]:
        raise ValueError("Protocol changed")
    protocol = json.loads(protocol_path.read_text("utf-8"))
    settings = protocol["robustness"]
    conditions = ["identity", "gain_minus_6_db", "white_noise_snr_20_db"]
    if (protocol["test_evaluation_allowed"] or settings["split"] != "validation"
            or settings["conditions"] != conditions or settings["seed"] != 20261028
            or settings["gain_amplitude"] != 10 ** (-6 / 20) or settings["clip_after_transform"]):
        raise ValueError("Unsupported or opened protocol")
    pointer = json.loads((ROOT / "reports/latest_candidate_inference_check.json").read_text("utf-8"))
    previous_path = ROOT / pointer["report"]
    if sha256_file(previous_path) != pointer["report_sha256"]:
        raise ValueError("Identity verification changed")
    previous = json.loads(previous_path.read_text("utf-8"))
    if (previous["status"] != "complete" or previous["tracks"] != 64 or previous["test_evaluated"]
            or previous["preparation_sha256"] != sha256_file(prepared_path)):
        raise ValueError("Identity prerequisite incomplete")
    for path, digest in previous["source_sha256"].items():
        if sha256_file(ROOT / path) != digest:
            raise ValueError(f"Identity prerequisite source changed: {path}")
    sample_path = ROOT / prepared["sample"]["path"]
    manifest_path = ROOT / prepared["baseline"]["folder"] / "manifest.csv"
    if (sha256_file(sample_path) != prepared["sample"]["sha256"]
            or sha256_file(manifest_path) != prepared["baseline"]["manifest_sha256"]):
        raise ValueError("Panel or manifest changed")
    cohort = read_csv(sample_path)
    manifest = {r["track_id"]: r for r in read_csv(manifest_path) if r["split"] == "validation"}
    if len(cohort) != 64 or len({r["track_id"] for r in cohort}) != 64 or len({r["artist_id"] for r in cohort}) != 64:
        raise ValueError("Incomplete panel")
    for row in cohort:
        if row["split"] != "validation" or any(manifest.get(row["track_id"], {}).get(k) != v for k, v in row.items()):
            raise ValueError("Panel is not the fixed validation selection")
    pin = prepared["runs"]["vit_block6_hpss"]
    reference_path = ROOT / pin["run"]["folder"] / "validation_predictions.csv"
    if sha256_file(reference_path) != pin["validation_predictions_sha256"]:
        raise ValueError("Reference predictions changed")
    reference = {r["track_id"]: r for r in read_csv(reference_path)}
    input_paths = [prepared_path, protocol_path, previous_path, sample_path, manifest_path, reference_path]
    hashes = {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in input_paths}
    for p in [*previous["source_sha256"], "src/music_genre/evaluation/robustness.py",
              "src/music_genre/evaluation/metrics.py", "src/music_genre/evaluation/errors.py",
              "scripts/evaluate_robustness.py"]:
        hashes[p] = sha256_file(ROOT / p)
    attempt = ROOT / "data/interim" / ("robustness-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    attempt.mkdir(parents=True)
    predictor = CandidatePredictor(ROOT)
    if predictor.model_id != previous["model_id"]:
        raise ValueError("Candidate identity changed")
    labels = predictor.labels
    if any(sum(r["genre"] == label for r in cohort) != 8 for label in labels):
        raise ValueError("Genre quotas changed")
    # One preselected track per genre; keep these results in the full panel.
    pilot = [next(r for r in cohort if r["genre"] == label) for label in labels]
    pilot_ids = {r["track_id"] for r in pilot}
    ordered = pilot + [r for r in cohort if r["track_id"] not in pilot_ids]
    results, benchmark = {}, None
    compute_started = perf_counter()
    with tempfile.TemporaryDirectory(dir=attempt, prefix="audio-") as temporary:
        temporary_path = Path(temporary) / "transformed.wav"
        for number, row in enumerate(ordered, 1):
            path = (ROOT / "data/raw" / row["audio_path"]).resolve()
            if not path.is_relative_to((ROOT / "data/raw/fma_small").resolve()) or sha256_file(path) != row["source_sha256"]:
                raise ValueError("Audio changed or escaped source directory")
            audio = load_audio(path, sample_rate=16000)
            original = audio.samples.copy()
            track = {"track_id": row["track_id"], "artist_id": row["artist_id"], "genre": row["genre"],
                     "source_sha256": row["source_sha256"], "conditions": {}}
            for condition in conditions:
                transformed = perturb_audio(original, condition, row["track_id"], seed=settings["seed"])
                if not np.array_equal(audio.samples, original) or not np.isfinite(transformed).all():
                    raise ValueError("Non-finite transformation or source mutation")
                sf.write(temporary_path, transformed, 16000, subtype="FLOAT")
                decoded, rate = sf.read(temporary_path, dtype="float32")
                if rate != 16000 or not np.array_equal(decoded, transformed):
                    raise ValueError("Temporary FLOAT WAV is not lossless")
                prediction = predictor.predict_file(temporary_path, mode="fma")
                probabilities = np.array([w["scores"] for w in prediction["windows"]])
                if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
                        or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)):
                    raise ValueError("Invalid window probabilities")
                windows = previous["results"]
                old = next(r["prediction"] for r in windows if r["track_id"] == row["track_id"])
                if [(w["start_seconds"], w["valid_seconds"], w["padded_seconds"]) for w in old["windows"]] != [
                        (w["start_seconds"], w["valid_seconds"], w["padded_seconds"]) for w in prediction["windows"]]:
                    raise ValueError("Window geometry changed")
                info = {"peak_abs": float(np.max(np.abs(transformed))),
                        "fraction_above_amplitude_one": float(np.mean(np.abs(transformed) > 1)),
                        "samples": len(transformed), "sample_rate": 16000}
                if condition == "identity":
                    expected = np.array([float(reference[row["track_id"]][f"p_{label}"]) for label in labels])
                    error = float(np.max(np.abs(expected - prediction["scores"])))
                    info["saved_prediction_error"] = error
                    if error > settings["hard_checks"]["maximum_identity_probability_error"]:
                        raise ValueError("Identity condition differs from saved prediction")
                if condition == "white_noise_snr_20_db":
                    noise = transformed.astype(np.float64) - original
                    snr = float(10 * np.log10(np.mean(original.astype(np.float64) ** 2) / np.mean(noise ** 2)))
                    info["measured_snr_db"] = snr
                    if abs(snr - 20) > 1e-4:
                        raise ValueError("Measured noise SNR differs from protocol")
                track["conditions"][condition] = {"audio": info, "prediction": prediction}
            if sha256_file(path) != row["source_sha256"]:
                raise ValueError("Original audio changed during run")
            results[row["track_id"]] = track
            write_json(attempt / f"{row['track_id']}.json", track)
            if number == 8:
                elapsed = perf_counter() - compute_started
                benchmark = {"tracks": 8, "conditions_per_track": 3, "track_ids": [r["track_id"] for r in pilot],
                             "elapsed_seconds": elapsed, "projected_panel_seconds": elapsed * 8,
                             "budget_seconds": 7200, "workers": 1, "results_reused": True}
                write_json(attempt / "benchmark.json", benchmark)
                print(json.dumps({"benchmark": benchmark}), flush=True)
                if benchmark["projected_panel_seconds"] > benchmark["budget_seconds"]:
                    raise RuntimeError("Pilot projection exceeds two-hour budget; review before continuing")
            state = {"status": "running", "completed_tracks": number, "total_tracks": 64,
                     "elapsed_seconds": perf_counter() - started, "benchmark": benchmark,
                     "attempt": attempt.relative_to(ROOT).as_posix(), "test_evaluated": False}
            write_json(PROGRESS, state)
            if number % 4 == 0:
                print(json.dumps(state), flush=True)
    if any(sha256_file(ROOT / p) != digest for p, digest in hashes.items()):
        raise ValueError("Inputs or source files changed during run")
    truth = np.array([labels.index(row["genre"]) for row in cohort])
    matrices = {c: np.array([results[r["track_id"]]["conditions"][c]["prediction"]["scores"] for r in cohort]) for c in conditions}
    summary = {}
    for condition in conditions:
        item = compare_condition(truth, matrices["identity"], matrices[condition], labels, settings["review_triggers"])
        item["paired_outcomes"] = paired_outcomes(truth, matrices["identity"].argmax(axis=1), matrices[condition].argmax(axis=1))
        item["flips_by_genre"] = {label: int(((truth == i) & (matrices["identity"].argmax(axis=1) != matrices[condition].argmax(axis=1))).sum()) for i, label in enumerate(labels)}
        summary[condition] = item
    report = {"status": "complete", "test_evaluated": False, "test_audio_loaded": False,
              "tracks": 64, "artists": 64, "conditions": conditions, "windows_per_condition": 192,
              "protocol_sha256": prepared["protocol_sha256"], "model_id": predictor.model_id,
              "source_and_input_sha256": hashes, "benchmark": benchmark,
              "elapsed_seconds": perf_counter() - started, "summary": summary,
              "review_required": any(any(v["review_triggers"].values()) for v in summary.values()),
              "versions": {p: version(p) for p in ("numpy", "scipy", "torch", "timm", "librosa", "soundfile", "scikit-learn")},
              "results": [results[r["track_id"]] for r in cohort]}
    report_path = ROOT / "reports/ROBUSTNESS_PANEL.json"
    write_json(report_path, report)
    write_json(ROOT / "reports/latest_robustness_panel.json", {"report": report_path.relative_to(ROOT).as_posix(), "report_sha256": sha256_file(report_path)})
    write_json(PROGRESS, {**state, "status": "complete", "review_required": report["review_required"],
                          "elapsed_seconds": report["elapsed_seconds"]})
    lines = ["# Sensibilité au volume et au bruit — panel de validation", "",
             ("64 morceaux, huit par genre et 64 artistes distincts. Ce panel fixé avant les essais "
             "n’est pas l’ensemble des 780 morceaux de validation. Test non évalué."), "",
             "| Condition | Macro-F1 | Accuracy | Perte F1 vs identité | Classes changées | Corrections / régressions | Revue déclenchée |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for condition, item in summary.items():
        pair = item["paired_outcomes"]
        lines.append(f"| {condition} | {item['metrics']['macro_f1']:.4f} | {item['metrics']['accuracy']:.4f} | "
                     f"{item['macro_f1_drop']:+.4f} | {item['flipped_tracks']}/64 | {pair['corrected']} / {pair['regressed']} | "
                     f"{'oui' if any(item['review_triggers'].values()) else 'non'} |")
    lines += ["", ("Seuils prévus : perte macro-F1 ≥ 0,05 ou changements de classe ≥ 20 %. "
              "Une revue déclenchée ne lance aucun ajustement du modèle."), "",
              "| Genre | F1 identité | F1 gain −6 dB | F1 bruit 20 dB | Changements gain / bruit |",
              "|---|---:|---:|---:|---:|"]
    for label in labels:
        values = [f"{summary[c]['metrics']['per_class'][label]['f1-score']:.4f}" for c in conditions]
        lines.append(f"| {label} | " + " | ".join(values) + f" | {summary[conditions[1]]['flips_by_genre'][label]} / {summary[conditions[2]]['flips_by_genre'][label]} |")
    lines += ["", ("Les fichiers d’origine sont conservés. Transformations sur mono 16 kHz avant "
              "segmentation, padding intact, passage WAV FLOAT vérifié sans perte, même API et "
              "même modèle. Ni normalisation supplémentaire ni écrêtage ajouté."), "",
              (f"Benchmark : projection {benchmark['projected_panel_seconds']/60:.2f} minutes sous le budget de 120 minutes. "
              f"Exécution complète : {report['elapsed_seconds']/60:.2f} minutes, un processus ; mesure indicative sur ce poste."), "",
              ("Avec huit morceaux par genre, ces scores sont descriptifs. Ils n’établissent pas une "
              "robustesse générale ni une cause acoustique certaine. Aucun intervalle de confiance "
              "ou test de significativité n’est revendiqué."), "",
              "Reproduction : `.\\.venv\\Scripts\\python.exe scripts/evaluate_robustness.py`.", "",
              ("[Résultats détaillés et empreintes](ROBUSTNESS_PANEL.json) · "
              "[Protocole](../docs/PROTOCOLE_EVALUATION_FINALE.md)"), ""]
    (ROOT / "reports/ROBUSTESSE_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"status": "complete", "summary": summary, "review_required": report["review_required"]}), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        previous_state = json.loads(PROGRESS.read_text("utf-8")) if PROGRESS.exists() else {}
        write_json(PROGRESS, {**previous_state, "status": "failed", "error": str(error), "test_evaluated": False})
        raise
