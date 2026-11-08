"""Validate saved CNN artifacts and compare systems on identical validation tracks."""

import csv
import json
from pathlib import Path

import numpy as np
from compare_vit_baseline import read_run

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.metrics import classification_metrics

ROOT = Path(__file__).resolve().parents[1]
REFERENCES = {
    "descriptor": "Descripteurs", "vit": "ViT gelé", "fusion": "Fusion complète",
    "fusion_no_rhythm": "Fusion sans rythme", "fusion_no_harmony": "Fusion sans harmonie",
}


def main():
    pointer = json.loads((ROOT / "reports/latest_cnn_run.json").read_text(encoding="utf-8"))
    folder = (ROOT / pointer["folder"]).resolve()
    if (not folder.is_relative_to((ROOT / "runs").resolve())
            or sha256_file(folder / "report.json") != pointer["report_sha256"]):
        raise ValueError("Invalid CNN report")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    if report["status"] not in ("complete", "budget_exhausted") or report["test_loaded"] or report["test_evaluated"]:
        raise ValueError("Expected a finished validation-only CNN experiment")
    for filename, key in (("model.safetensors", "model_sha256"),
                          ("validation_predictions.csv", "validation_predictions_sha256")):
        if sha256_file(folder / filename) != report[key]:
            raise ValueError(f"CNN artifact changed: {filename}")
    cache = (ROOT / report["feature_cache"]).resolve()
    if (not cache.is_relative_to((ROOT / "data/features").resolve())
            or sha256_file(cache / "report.json") != report["feature_report_sha256"]):
        raise ValueError("CNN cache report changed")
    features = json.loads((cache / "report.json").read_text(encoding="utf-8"))
    for filename, key in (("features.npy", "features_sha256"), ("index.csv", "index_sha256")):
        if sha256_file(cache / filename) != features[key]:
            raise ValueError(f"CNN cache changed: {filename}")
    if (features["index_sha256"] != report["index_sha256"]
            or features["provenance"]["manifest_sha256"] != report["baseline_manifest_sha256"]):
        raise ValueError("CNN index or manifest mismatch")
    with (cache / "index.csv").open(encoding="utf-8", newline="") as stream:
        index = list(csv.DictReader(stream))
    labels = report["labels"]
    expected = {}
    for row in index:
        if row["split"] not in ("train", "validation"):
            raise ValueError("Unexpected test window")
        if row["split"] == "validation":
            label = labels[int(row["label_index"])]
            if expected.setdefault(row["track_id"], label) != label:
                raise ValueError("Inconsistent track label")
    with (folder / "validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    keys = [(r["track_id"], r["true_label"]) for r in predictions]
    if keys != sorted(expected.items()):
        raise ValueError("Incomplete or misaligned CNN predictions")
    truth = np.array([labels.index(r["true_label"]) for r in predictions])
    probabilities = np.array([[float(r[f"p_{label}"]) for label in labels] for r in predictions])
    if (not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.allclose(probabilities.sum(axis=1), 1)):
        raise ValueError("Invalid CNN probabilities")
    metrics = classification_metrics(truth, probabilities, labels)
    if metrics != report["validation"]:
        raise ValueError("Recomputed CNN metrics differ")
    history = report["history"]
    if ([r["epoch"] for r in history] != list(range(1, report["epochs_completed"] + 1))
            or not all(np.isfinite(r["training_window_loss"]) for r in history)):
        raise ValueError("Invalid epoch history")
    best = max(history, key=lambda r: r["validation"]["macro_f1"])
    if best["epoch"] != report["selected_epoch"] or best["validation"] != metrics:
        raise ValueError("CNN epoch selection differs from its declared criterion")
    refs = {name: read_run(name) for name in REFERENCES}
    for _, reference, feature_report, rows, _ in refs.values():
        for key in ("baseline_manifest_sha256", "labels", "track_counts", "window_counts"):
            if reference[key] != report[key]:
                raise ValueError(f"Incomparable CNN/reference: {key}")
        if feature_report["index_sha256"] != report["index_sha256"]:
            raise ValueError("Reference windows differ")
        if [(r["track_id"], r["true_label"]) for r in rows] != keys:
            raise ValueError("Reference predictions differ in order or labels")
        if reference["test_loaded"] or reference["test_evaluated"]:
            raise ValueError("Reference used test")
    vit = refs["vit"][1]
    correct = probabilities.argmax(axis=1) == truth
    vit_correct = refs["vit"][4]
    comparison = {
        "cnn_run": pointer, "reference_runs": {k: v[0] for k, v in refs.items()},
        "index_sha256": report["index_sha256"], "selected_epoch": report["selected_epoch"],
        "macro_f1_difference_cnn_minus_vit": metrics["macro_f1"] - vit["validation"]["macro_f1"],
        "accuracy_difference_cnn_minus_vit": metrics["accuracy"] - vit["validation"]["accuracy"],
        "cnn_correct_vit_wrong": int((correct & ~vit_correct).sum()),
        "vit_correct_cnn_wrong": int((~correct & vit_correct).sum()),
        "test_evaluated": False,
    }
    write_json(ROOT / "reports/CNN_COMPARISON.json", comparison)
    lines = [
        "# CNN de référence — entraînement et comparaison", "",
        (f"{report['parameters']:,} paramètres, apprentissage de zéro, graine {report['config']['seed']}. "
         f"{report['epochs_completed']} époques réalisées ; époque **{report['selected_epoch']}** retenue "
         "sur la macro-F1 par morceau. Même sélection filtrée : 6 299 morceaux de train et 780 de validation. "
         "Le test n’est pas évalué."), "",
        f"Motif d’arrêt enregistré : `{report['stop_reason']}`. Statut : `{report['status']}`.", "",
        "## Comparaison par morceau", "",
        "| Système | Macro-F1 validation | Accuracy validation | Macro-F1 train |",
        "|---|---:|---:|---:|",
    ]
    for name, result in [(REFERENCES[k], v[1]) for k, v in refs.items()] + [("CNN de zéro", report)]:
        lines.append(f"| {name} | {result['validation']['macro_f1']:.4f} | "
                     f"{result['validation']['accuracy']:.4f} | {result['training']['macro_f1']:.4f} |")
    lines += ["",
              (f"Écart CNN − ViT : **{comparison['macro_f1_difference_cnn_minus_vit']:+.4f}** en macro-F1 "
               f"et **{comparison['accuracy_difference_cnn_minus_vit']:+.4f}** en accuracy."),
              ("Le classement principal repose sur la macro-F1, fixée au plan : chaque genre y a le même poids. "
               "Une meilleure accuracy n’implique pas une meilleure macro-F1, car les erreurs "
               "peuvent être réparties différemment entre genres."), "",
              "## Historique complet", "",
              "| Époque | Loss train par fenêtre | Macro-F1 validation | Accuracy validation | Durée (s) |",
              "|---:|---:|---:|---:|---:|"]
    for row in history:
        lines.append(f"| {row['epoch']} | {row['training_window_loss']:.4f} | "
                     f"{row['validation']['macro_f1']:.4f} | {row['validation']['accuracy']:.4f} | {row['seconds']:.1f} |")
    lines += ["",
              ("La loss est mesurée pendant les mises à jour sur les fenêtres de train ; les scores "
               "de validation sont calculés ensuite, par morceau, en mode évaluation. Le score train "
               "du tableau principal est recalculé avec les poids de l’époque retenue."), "",
              "## Résultats par genre", "", "| Genre | Effectif | F1 ViT | F1 CNN | Rappel CNN |",
              "|---|---:|---:|---:|---:|"]
    for label in labels:
        cnn_class, vit_class = metrics["per_class"][label], vit["validation"]["per_class"][label]
        lines.append(f"| {label} | {int(cnn_class['support'])} | {vit_class['f1-score']:.4f} | "
                     f"{cnn_class['f1-score']:.4f} | {cnn_class['recall']:.4f} |")
    lines += ["", "## Confusion CNN", "", "Lignes : genre réel ; colonnes : genre prédit.", "",
              "| Réel / prédit | " + " | ".join(labels) + " |", "|---|" + "---:|" * len(labels)]
    for label, row in zip(labels, metrics["confusion_matrix"], strict=True):
        lines.append("| " + label + " | " + " | ".join(str(n) for n in row) + " |")
    lines += ["",
              (f"CNN correct et ViT incorrect : {comparison['cnn_correct_vit_wrong']} morceaux ; "
               f"ViT correct et CNN incorrect : {comparison['vit_correct_cnn_wrong']}."), "",
              "## Coût et vérifications", "",
              f"- Extraction du cache log-Mel : {features['elapsed_seconds'] / 60:.2f} min.",
              f"- Entraînement et validation : {report['training_validation_seconds'] / 60:.2f} min.",
              f"- Avec export et évaluation finale du train : {report['elapsed_including_export_and_train_evaluation_seconds'] / 60:.2f} min.",
              f"- Écart maximal entre probabilités avant/après export : {report['export_max_probability_error']:.3g}.", "",
              ("Rapports, matrice log-Mel, index, poids et prédictions vérifiés par empreintes. "
               "Même index de fenêtres pour toutes les références. Métriques recalculées depuis les "
               "probabilités exportées ; meilleure époque revérifiée sur tout l’historique."), "",
              "## Limites et suite", "",
              ("Ce CNN est une référence de développement à une seule graine, limitée à dix époques. "
               "Le motif max_epochs ne démontre pas une convergence ou une performance maximale. "
               "Le benchmark CPU a été respecté sans ouvrir le jeu de test."), "",
              ("CNN : log-Mel natif, apprentissage de zéro, sélection d’époque. ViT : log-Mel redimensionné, "
               "poids image préentraînés et gelés, sélection de C. Les budgets de sélection diffèrent ; "
               "c’est une comparaison de systèmes, pas une ablation de l’architecture seule. "
               "La validation sert au développement ; aucun écart n’est présenté comme statistiquement établi."), "",
              ("Prochain jalon : ablation ViT fenêtre centrale seule contre trois fenêtres, en réutilisant "
               "les embeddings existants, avec les mêmes morceaux et la même grille C. Les autres ablations, "
               "répétitions stochastiques, verrouillage du test et démo restent à réaliser."), "",
              "## Reproduction et preuves", "", "Voir [le protocole CNN](../docs/CNN.md).", "",
              f"- [Rapport CNN complet](../{pointer['folder']}/report.json)",
              f"- [Modèle exporté](../{pointer['folder']}/model.safetensors)",
              "- [Comparaison vérifiable](CNN_COMPARISON.json)", ""]
    (ROOT / "reports/BASELINE_CNN.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
