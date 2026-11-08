"""Compare temporal disagreements on all validation tracks using fixed exports."""

import csv
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from analyze_validation_errors import SYSTEMS, load_inputs
from diagnose_review_audio import load_cache, read_json
from threadpoolctl import threadpool_limits

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.temporal import analyze_temporal
from music_genre.models.linear import LinearProbabilityModel

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["vit", "vit_block6", "vit_hpss"]


def plot_counts(folder, results):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/interim/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), layout="constrained", sharey=True)
    for ax, name in zip(axes, MODELS, strict=True):
        table = np.array(results[name]["all"]["correct_window_count_vs_mean"])
        ax.bar(range(4), table[:, 0], color="#bc6046", label="Moyenne incorrecte")
        ax.bar(range(4), table[:, 1], bottom=table[:, 0], color="#237c75", label="Moyenne correcte")
        for k in range(4):
            for col in range(2):
                if table[k, col]:
                    y = table[k, :col].sum() + table[k, col] / 2
                    ax.text(k, y, str(table[k, col]), ha="center", va="center", color="white")
        ax.set(title=SYSTEMS[name], xlabel="Fenêtres correctement classées sur 3", xticks=range(4))
    axes[0].set_ylabel("Nombre de morceaux")
    axes[-1].legend(loc="upper right", fontsize=8)
    fig.suptitle("780 morceaux de validation · modèles fixes\nLa moyenne peut perdre ou récupérer une prédiction correcte")
    paths = []
    for ext in ("png", "pdf"):
        path = folder / f"temporal_counts.{ext}"
        fig.savefig(path, dpi=150)
        paths.append({"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)})
    plt.close(fig)
    return paths


def write_report(result):
    models, labels = result["models"], result["labels"]
    lines = ["# Désaccords temporels sur toute la validation", "",
             ("780 morceaux, 2 340 fenêtres par modèle, trois ViT avec classificateurs exportés inchangés. "
             "Aucun entraînement, nouvelle extraction audio ou accès aux données de test."), "",
             "## Désaccords et agrégation", "",
             "| Modèle | Désaccord entre fenêtres | Erreurs de la moyenne | Erreurs avec ≥1 fenêtre correcte | Moyenne correcte sans fenêtre correcte | Majorité unique différente de la moyenne | Sans majorité unique |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in models.items():
        c = r["all"]
        lines.append(f"| {SYSTEMS[name]} | {c['disagreements']}/780 | {c['aggregate_wrong']} | "
                     f"{c['wrong_with_correct_window']} | {c['correct_with_no_correct_window']} | "
                     f"{c['unique_majority_differs_from_mean']} | {c['no_unique_majority']} |")
    lines += ["", ("Un désaccord signifie que les trois argmax ne sont pas identiques. "
              "Une majorité unique exige au moins deux votes identiques ; trois genres distincts n’en ont pas. "
              "Les égalités d’argmax suivent l’ordre des classes. Aucun score d’un nouveau vote majoritaire n’est sélectionné."), "",
              f"![Comptages par nombre de fenêtres correctes](../{result['figures'][0]['path']})", "",
              "## Même classificateur, passages différents", "",
              "| Modèle | Entrée évaluée | Macro-F1 | Accuracy |", "|---|---|---:|---:|"]
    for name, r in models.items():
        for title, m in [("Moyenne des trois", r["mean_metrics"]),
                         *[(f"{10*w}–{10*(w+1)} s", m) for w, m in enumerate(r["window_metrics"])]]:
            lines.append(f"| {SYSTEMS[name]} | {title} | {m['macro_f1']:.4f} | {m['accuracy']:.4f} |")
    lines += ["", ("Ces évaluations par position conservent les modèles appris sur trois fenêtres. "
              "Elles ne sont ni des réentraînements sur une seule fenêtre ni une sélection de la meilleure position."), "",
              "## Détail par genre", "",
              "| Modèle | Genre FMA | Morceaux | Artistes | Désaccords | Erreurs moyenne | Erreurs avec fenêtre correcte | Moyenne correcte sans fenêtre correcte |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for name, r in models.items():
        for label in labels:
            c = r["by_genre"][label]
            lines.append(f"| {SYSTEMS[name]} | {label} | {c['tracks']} | {c['artists']} | "
                         f"{c['disagreements']} | {c['aggregate_wrong']} | {c['wrong_with_correct_window']} | "
                         f"{c['correct_with_no_correct_window']} |")
    lines += ["", "## Contrôles et limites", "",
              ("Les trois caches, modèles, prédictions enregistrées, sources d’extraction et versions sont vérifiés. "
              "Chaque morceau correspond exactement au manifeste de validation, avec les fenêtres 0/10/20 s, "
              "le même artiste, label, fichier source et minutage entre caches. Les moyennes reproduisent "
              "les probabilités et métriques historiques."), "",
              (f"Écart maximal des probabilités : {max(r['max_saved_probability_error'] for r in models.values()):.3g}. "
              f"Durée totale du diagnostic : {result['elapsed_seconds']:.2f} s."), "",
              ("La présence d’une fenêtre correcte utilise la vérité terrain : c’est une description a posteriori, "
              "pas un sélecteur utilisable à l’inférence ni un gain atteignable démontré. Des désaccords ne prouvent "
              "pas un changement réel de genre, un défaut de label ou un problème de prétraitement. "
              "Les morceaux d’un artiste ne sont pas indépendants ; aucun intervalle de confiance ni test de significativité n’est revendiqué. "
              "La validation reste un jeu de développement. Aucune écoute subjective n’a été réalisée."), "",
              ("[Protocole et interprétation](../docs/ANALYSE_TEMPORELLE.md) · "
              "[Rapport numérique et probabilités](TEMPORAL_ANALYSIS.json)"), "",
              "Commande : `.venv/Scripts/python.exe scripts/analyze_temporal_validation.py`.", ""]
    (ROOT / "reports/ANALYSE_TEMPORELLE.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    started = perf_counter()
    config_path = ROOT / "configs/temporal_analysis.json"
    config = read_json(config_path)
    if (config["split"] != "validation" or config["models"] != MODELS
            or config["offsets_seconds"] != [0., 10., 20.]
            or config["aggregation"] != "arithmetic_mean_of_three_window_probabilities"):
        raise ValueError("Unsupported temporal protocol")
    tolerance = config["probability_absolute_tolerance"]
    if not np.isfinite(tolerance) or not 0 <= tolerance <= 1e-5:
        raise ValueError("Invalid comparison tolerance")
    baseline, sources, labels, ids, truth, saved, runs, pointers = load_inputs()
    if len(ids) != config["expected_tracks"]:
        raise ValueError("Validation cohort size changed")
    results, caches, reference_index = {}, {}, None
    with threadpool_limits(limits=1):
        for name in MODELS:
            print(f"Verifying cache and fixed model: {name}", flush=True)
            pointer, _report, indexed, matrix = load_cache(name, baseline, labels)
            if (pointer["folder"] != runs[name]["feature_cache"]
                    or pointer["report_sha256"] != runs[name]["feature_report_sha256"]):
                raise ValueError("Cache differs from trained model")
            rows = [row for row in indexed.values() if row["split"] == "validation"]
            if {r["track_id"] for r in rows} != set(ids) or len(rows) != 3 * len(ids):
                raise ValueError("Missing or extra validation windows")
            for row in rows:
                source = sources[row["track_id"]]
                if any(row[k] != source[k] for k in ("split", "artist_id", "label_index", "source_sha256")):
                    raise ValueError("Window metadata differs from manifest")
            identity = sorted(tuple(row[k] for k in (
                "track_id", "window_id", "start_seconds", "valid_seconds", "padded_seconds",
                "split", "label_index", "artist_id", "source_sha256")) for row in rows)
            if reference_index is not None and identity != reference_index:
                raise ValueError("Temporal windows differ between models")
            reference_index = identity
            model = LinearProbabilityModel.load(ROOT / pointers[name]["folder"] / "model.npz")
            if not np.array_equal(model.classes, np.arange(len(labels))):
                raise ValueError("Class order differs")
            probabilities = model.predict_proba(matrix[[int(row["row_index"]) for row in rows]])
            r = analyze_temporal(rows, probabilities, labels)
            if [t["track_id"] for t in r["tracks"]] != ids:
                raise ValueError("Track order differs")
            if [t["true_label"] for t in r["tracks"]] != [labels[int(y)] for y in truth]:
                raise ValueError("Track labels differ")
            error = float(np.max(np.abs(np.array([t["mean_probabilities"] for t in r["tracks"]]) - saved[name])))
            if not np.isfinite(error) or error > tolerance or r["mean_metrics"] != runs[name]["validation"]:
                raise ValueError("Saved predictions or metrics differ")
            r["max_saved_probability_error"] = error
            results[name], caches[name] = r, pointer
            print(json.dumps({"model": name, **r["all"]}), flush=True)
    folder = ROOT / "reports/figures" / ("temporal-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    folder.mkdir(parents=True, exist_ok=False)
    figures = plot_counts(folder, results)
    csv_path = folder / "track_counts.csv"
    fields = ["track_id", "artist_id", "true_label", "aggregate_label", "majority_label",
              "correct_windows", "aggregate_correct", "disagreement"]
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["model", *fields, "window_0", "window_1", "window_2"])
        writer.writeheader()
        for name, r in results.items():
            for t in r["tracks"]:
                writer.writerow({"model": name, **{k: t[k] for k in fields},
                                 **{f"window_{w}": t["window_labels"][w] for w in range(3)}})
    result = {"schema_version": 1, "status": "complete", "settings": config,
              "baseline": baseline, "labels": labels, "models": results, "caches": caches,
              "runs": {name: pointers[name] for name in MODELS}, "figures": figures,
              "track_csv": {"path": csv_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(csv_path)},
              "config_sha256": sha256_file(config_path), "test_evaluated": False,
              "test_features_loaded": False, "audio_decoded": False, "models_refit": False,
              "elapsed_seconds": perf_counter() - started,
              "source_sha256": {p: sha256_file(ROOT / p) for p in (
                  "scripts/analyze_temporal_validation.py", "scripts/analyze_validation_errors.py",
                  "scripts/diagnose_review_audio.py", "src/music_genre/evaluation/temporal.py",
                  "src/music_genre/evaluation/metrics.py", "src/music_genre/evaluation/errors.py",
                  "src/music_genre/models/linear.py", "src/music_genre/data/audit.py")}}
    write_json(ROOT / "reports/TEMPORAL_ANALYSIS.json", result)
    write_json(ROOT / "reports/latest_temporal_analysis.json", {
        "report": "reports/TEMPORAL_ANALYSIS.json", "report_sha256": sha256_file(ROOT / "reports/TEMPORAL_ANALYSIS.json")})
    write_report(result)
    print("Complete: 780 validation tracks, 2340 windows per model, three fixed models", flush=True)


if __name__ == "__main__":
    main()
