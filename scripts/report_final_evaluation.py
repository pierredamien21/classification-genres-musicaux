"""Report a completed validation/test campaign using the locked statistical protocol."""

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.final_evaluation import PredictionStore, json_digest
from music_genre.evaluation.final_protocol import (
    inside,
    load_locked_manifest,
    validate_evaluation_request,
)
from music_genre.evaluation.final_reporting import summarize_predictions

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(path.read_text("utf-8"))


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def markdown(report):
    split = report["split"]
    lines = [f"# Évaluation — {split}", "", "Statistiques calculées ; contrôle indépendant du rapport restant à effectuer.",
             "", f"{report['tracks']} morceaux, {report['artists']} artistes, huit labels fixes.",
             "", "| Système | Macro-F1 [IC 95 %] | Accuracy [IC 95 %] |",
             "|---|---:|---:|"]
    for name, values in report["estimates"].items():
        cells = [f"{values[m]['point']:.4f} [{values[m]['low']:.4f} ; {values[m]['high']:.4f}]"
                 for m in ("macro_f1", "accuracy")]
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += ["", "Intervalles percentile ponctuels : 2 000 tirages appariés d’artistes,",
              "conditionnels aux modèles fixés. Ils ne couvrent pas toute la variabilité",
              "de l’apprentissage et ne sont pas des tests simultanés de supériorité.", "",
              "La ligne cnn20_mean moyenne les métriques de trois graines, sans ensemble",
              "de probabilités. Leur dispersion entre graines est enregistrée séparément.", "",
              "## Comparaisons annoncées", "", "| Différence gauche − droite | Δ macro-F1 [IC 95 %] | Δ accuracy [IC 95 %] |",
              "|---|---:|---:|"]
    for name, values in report["differences"].items():
        cells = [f"{values[m]['point']:+.4f} [{values[m]['low']:+.4f} ; {values[m]['high']:+.4f}]"
                 for m in ("macro_f1", "accuracy")]
        lines.append(f"| {name} | {' | '.join(cells)} |")
    lines += ["", "## Candidat par genre", "", "| Genre | Précision | Rappel | F1 | Effectif |",
              "|---|---:|---:|---:|---:|"]
    candidate = report["candidate"]
    for label in report["labels"]:
        row = report["classification"][candidate]["per_class"][label]
        lines.append(f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {int(row['support'])} |")
    lines += ["", "## Coût et limites", "",
              "Les durées par appel proviennent des prédictions sauvegardées : elles incluent",
              "le prétraitement et l’inférence, mais pas tous les contrôles de campagne.",
              "La durée de la dernière invocation n’est pas une somme de reprises.", "",
              "Limites conservées : huit genres fermés, pas de gospel ni de détection du",
              "genre inconnu, scores non calibrés, fragilité au bruit, écoute subjective absente."]
    if split == "validation":
        lines += ["", "**Validation utilisée pour sélectionner les modèles : aucune performance indépendante",
                  "de généralisation n’est revendiquée dans ce rapport.**"]
    lines += ["", "Les matrices de confusion sont dans `confusions.csv`, les détails par classe",
              "dans `per_class.csv`, les preuves dans `report.json` et les tirages dans",
              "`bootstrap_draws.npz`. Les lignes sont les vrais genres, les colonnes les prédictions."]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    folder, output = inside(ROOT, args.campaign), inside(ROOT, args.output)
    if output == ROOT / "reports" or not output.is_relative_to(ROOT / "reports") or output.exists():
        raise ValueError("Choose a new report directory within reports")
    started = perf_counter()
    completion = read(folder / "completion.json")
    campaign = read(folder / "campaign.json")
    progress = read(folder / "progress.json")
    split = campaign["split"]
    expected_status = {"validation": "validation_passed", "test": "inference_complete"}
    if (split not in expected_status or completion["status"] != expected_status[split]
            or progress["status"] != "complete"
            or progress["completion_sha256"] != sha256_file(folder / "completion.json")
            or completion["campaign_sha256"] != json_digest(campaign)):
        raise ValueError("Campaign incomplete or changed")
    ref = completion["lock"]
    locked = load_locked_manifest(ROOT, ref["report"], ref["report_sha256"])
    if campaign["lock_sha256"] != ref["report_sha256"] or campaign["labels"] != locked["labels"]:
        raise ValueError("Campaign and locked protocol differ")
    sources = [Path(__file__).relative_to(ROOT).as_posix(),
               "src/music_genre/evaluation/final_reporting.py",
               "src/music_genre/evaluation/final_evaluation.py",
               "src/music_genre/evaluation/final_protocol.py",
               "src/music_genre/evaluation/bootstrap.py", "src/music_genre/evaluation/metrics.py"]
    evidence = {p: sha256_file(inside(ROOT, p)) for p in sources}
    for name, digest in campaign["runner_sources"].items():
        if sha256_file(inside(ROOT, name)) != digest:
            raise ValueError("Inference source changed")
        evidence[name] = digest
    for name in ("campaign.json", "completion.json"):
        path = folder / name
        evidence[path.relative_to(ROOT).as_posix()] = sha256_file(path)
    expected_ids = {}
    for stage in ("T4", "T5"):
        expected_ids.update({n: v["model_id"] for n, v in
                             read(inside(ROOT, locked["proofs"][stage]))["systems"].items()})
    if campaign["models"] != expected_ids:
        raise ValueError("Models differ from verified exports")
    manifest = csv_rows(inside(ROOT, locked["dataset"]["manifest"]))
    rows = sorted([r for r in manifest if r["split"] == split], key=lambda r: r["track_id"])
    validate_evaluation_request(locked, [r["track_id"] for r in rows], split)
    if (campaign["track_ids"] != [r["track_id"] for r in rows]
            or campaign["source_sha256"] != {r["track_id"]: r["source_sha256"] for r in rows}):
        raise ValueError("Prediction cohort differs from manifest")
    store = PredictionStore(folder, campaign)
    if store.receipts() != completion["predictions"]:
        raise ValueError("Prediction fingerprints differ")
    predictions, timings = {}, {}
    for name in campaign["models"]:
        predictions[name], durations = [], []
        for row in rows:
            item = store.read(name, row["track_id"])
            predictions[name].append({"track_id": row["track_id"], "scores": item["scores"]})
            durations.append(item["elapsed_seconds"])
        if not np.isfinite(durations).all() or min(durations) < 0:
            raise ValueError("Invalid saved inference durations")
        timings[name] = {"sum_call_seconds": float(sum(durations)),
                         "median_call_seconds": float(np.median(durations)),
                         "p95_call_seconds": float(np.quantile(durations, .95))}
    tracks = [{"track_id": r["track_id"], "artist_id": r["artist_id"],
               "label_index": int(r["label_index"])} for r in rows]
    counts = np.bincount([int(r["label_index"]) for r in manifest if r["split"] == "train"], minlength=8)
    with threadpool_limits(limits=1):
        result, arrays = summarize_predictions(tracks, predictions, locked["labels"], locked["protocol"], counts)
    equivalence = "not_applicable_to_test"
    if split == "validation":
        prior_path = inside(ROOT, locked["proofs"]["T3"])
        prior = read(prior_path)
        for field in ("models", "artist_ids", "estimates", "differences", "missing_class_replicates"):
            if result[field] != prior[field]:
                raise ValueError(f"Validation report differs from prior verified bootstrap: {field}")
        with np.load(inside(ROOT, prior["archive"]["path"]), allow_pickle=False) as archived:
            if any(not np.array_equal(value, archived[key]) for key, value in arrays.items()):
                raise ValueError("Validation resampling arrays differ from prior verified bootstrap")
        equivalence = "exact_match_to_T3_points_intervals_and_resampling_arrays"
    if any(sha256_file(inside(ROOT, p)) != h for p, h in evidence.items()):
        raise ValueError("Reporting source or evidence changed")
    output.mkdir(parents=True)
    archive = output / "bootstrap_draws.npz"
    np.savez_compressed(archive, **arrays)
    result.update({"status": "complete", "independent_report_check": "pending",
                   "split": split, "candidate": locked["protocol"]["candidate"],
                   "created_at": datetime.now(UTC).isoformat(), "lock": ref,
                   "campaign_path": args.campaign, "completion_sha256": sha256_file(folder / "completion.json"),
                   "source_and_input_sha256": evidence, "inference_timings": timings,
                   "last_inference_invocation_seconds": completion["last_invocation_seconds"],
                   "validation_equivalence": equivalence, "audio_loaded": False,
                   "test_evaluated": split == "test", "elapsed_seconds": perf_counter() - started,
                   "archive": {"path": archive.relative_to(ROOT).as_posix(), "sha256": sha256_file(archive)},
                   "limitations": ["fixed_model_conditional_pointwise_intervals", "not_simultaneous_tests",
                                   "cnn_mean_of_metrics_not_probability_ensemble", "no_seed_resampling",
                                   "no_unknown_genre_detection", "noise_sensitivity", "uncalibrated_scores",
                                   "subjective_listening_not_performed"]})
    with (output / "confusions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["system", "true_label", *result["labels"]])
        for name, metrics in result["classification"].items():
            for label, row in zip(result["labels"], metrics["confusion_matrix"], strict=True):
                writer.writerow([name, label, *row])
    with (output / "per_class.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["system", "label", "precision", "recall", "f1", "support"])
        for name, metrics in result["classification"].items():
            for label in result["labels"]:
                r = metrics["per_class"][label]
                writer.writerow([name, label, r["precision"], r["recall"], r["f1-score"], r["support"]])
    (output / "REPORT.md").write_text(markdown(result), encoding="utf-8")
    result["exports_sha256"] = {p.name: sha256_file(p) for p in output.iterdir() if p.is_file()}
    write_json(output / "report.json", result)
    print(json.dumps({"status": "complete", "split": split, "report": str(output / "report.json"),
                      "validation_equivalence": equivalence}), flush=True)


if __name__ == "__main__":
    main()
