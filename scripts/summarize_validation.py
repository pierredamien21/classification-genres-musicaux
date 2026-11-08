"""Consolidate verified development predictions; never load held-out test data."""

import csv
import json
from pathlib import Path

import numpy as np
from summarize_cnn_extension import read_extension
from summarize_cnn_repetitions import read_repetitions

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.metrics import classification_metrics

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = {
    "descriptor": "Descripteurs musicaux",
    "vit": "ViT bloc 12, trois fenêtres",
    "fusion": "Fusion bloc 12 + descripteurs",
    "fusion_no_rhythm": "Fusion sans rythme",
    "fusion_no_harmony": "Fusion sans harmonie",
    "cnn": "CNN, graine 42",
    "vit_central": "ViT bloc 12, fenêtre centrale",
    "vit_block6": "ViT bloc 6, trois fenêtres",
    "vit_hpss": "ViT bloc 12, canaux HPSS",
    "vit_block6_hpss": "ViT bloc 6, canaux HPSS",
}


def main():
    results, common = {}, None
    for name, label in SYSTEMS.items():
        pointer = json.loads((ROOT / f"reports/latest_{name}_run.json").read_text(encoding="utf-8"))
        folder = (ROOT / pointer["folder"]).resolve()
        if (not folder.is_relative_to((ROOT / "runs").resolve())
                or sha256_file(folder / "report.json") != pointer["report_sha256"]):
            raise ValueError(f"Invalid report: {name}")
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        if report["test_loaded"] or report["test_evaluated"]:
            raise ValueError("Only validation-only experiments are permitted")
        model_name = "model.safetensors" if name == "cnn" else "model.npz"
        for filename, key in ((model_name, "model_sha256"),
                              ("validation_predictions.csv", "validation_predictions_sha256")):
            if sha256_file(folder / filename) != report[key]:
                raise ValueError(f"Changed artifact: {name}/{filename}")
        with (folder / "validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        labels = report["labels"]
        truth = np.array([labels.index(r["true_label"]) for r in rows])
        probs = np.array([[float(r[f"p_{genre}"]) for genre in labels] for r in rows])
        if (not np.isfinite(probs).all() or np.any(probs < 0)
                or not np.allclose(probs.sum(axis=1), 1)):
            raise ValueError("Invalid probabilities")
        metrics = classification_metrics(truth, probs, labels)
        if metrics != report["validation"]:
            raise ValueError("Validation metrics cannot be reproduced")
        identity = {
            "manifest_sha256": report["baseline_manifest_sha256"], "labels": labels,
            "track_counts": report["track_counts"],
            "validation_tracks_and_truth": [(r["track_id"], r["true_label"]) for r in rows],
        }
        if common is None:
            common = identity
        if identity != common or len({r["track_id"] for r in rows}) != len(rows):
            raise ValueError("Incomparable or duplicated validation tracks")
        results[name] = {
            "label": label, "run": pointer, "validation": metrics,
            "training_macro_f1": report["training"]["macro_f1"],
            "selection": {"epoch": report["selected_epoch"], "seed": report["config"]["seed"]}
            if name == "cnn" else {"c": report["selected_c"]},
            "window_counts": report["window_counts"],
        }
        if name == "descriptor":
            majority = report["majority_validation"]
    repetition_pointer = ROOT / "reports/latest_cnn_repetitions.json"
    repetition_summary = None
    if repetition_pointer.exists():
        pointer = json.loads(repetition_pointer.read_text(encoding="utf-8"))
        path = (ROOT / pointer["report"]).resolve()
        if (not path.is_relative_to((ROOT / "reports").resolve())
                or sha256_file(path) != pointer["report_sha256"]):
            raise ValueError("Repetition summary changed")
        saved = json.loads(path.read_text(encoding="utf-8"))
        plan, pointers, cnn_reports, repetition_summary = read_repetitions()
        if (saved["summary"] != repetition_summary or saved["runs"] != pointers
                or saved["plan_sha256"] != sha256_file(ROOT / "configs/cnn_repetitions.json")
                or pointers[0] != results["cnn"]["run"]):
            raise ValueError("Repetition summary does not match the verified runs")
        results["cnn"] = {
            "label": "CNN, moyenne de trois graines", "runs": pointers,
            "validation": {name: stats["mean"] for name, stats in repetition_summary.items()},
            "validation_sample_std": {name: stats["sample_std"] for name, stats in repetition_summary.items()},
            "training_macro_f1": float(np.mean([r["training"]["macro_f1"] for r in cnn_reports])),
            "selection": {"seeds": plan["seeds"]}, "window_counts": cnn_reports[0]["window_counts"],
            "aggregation": "mean_of_per_seed_metrics_not_an_ensemble",
        }
    extension_complete = False
    if (ROOT / "reports/latest_cnn_extension.json").exists():
        pointer = json.loads((ROOT / "reports/latest_cnn_extension.json").read_text(encoding="utf-8"))
        path = (ROOT / pointer["report"]).resolve()
        if (not path.is_relative_to((ROOT / "reports").resolve())
                or sha256_file(path) != pointer["report_sha256"]):
            raise ValueError("Extension summary changed")
        saved = json.loads(path.read_text(encoding="utf-8"))
        plan, pointers, _, cnn_reports, extension_summary, differences = read_extension()
        if (saved["summary"] != extension_summary or saved["runs"] != pointers
                or saved["paired_differences"] != differences
                or saved["plan_sha256"] != sha256_file(ROOT / "configs/cnn_extension.json")):
            raise ValueError("Extension summary does not match verified runs")
        results["cnn20"] = {
            "label": "CNN, plafond 20, moyenne de trois graines", "runs": pointers,
            "validation": {name: stats["mean"] for name, stats in extension_summary.items()},
            "validation_sample_std": {name: stats["sample_std"] for name, stats in extension_summary.items()},
            "training_macro_f1": float(np.mean([r["training"]["macro_f1"] for r in cnn_reports])),
            "selection": {"seeds": plan["seeds"]}, "window_counts": cnn_reports[0]["window_counts"],
            "aggregation": "mean_of_per_seed_metrics_not_an_ensemble",
        }
        extension_complete = True
    ordered = sorted(results, key=lambda name: results[name]["validation"]["macro_f1"], reverse=True)
    best = ordered[0]
    output = {
        "schema_version": 1, "systems": results, "order_by_validation_macro_f1": ordered,
        "best_development_system": best, "majority_validation": majority,
        "manifest_sha256": common["manifest_sha256"], "track_counts": common["track_counts"],
        "labels": common["labels"], "test_evaluated": False,
        "cnn_three_seed_repetitions_complete": repetition_summary is not None,
        "cnn_twenty_epoch_extension_complete": extension_complete,
        "source_sha256": sha256_file(Path(__file__)),
    }
    write_json(ROOT / "reports/VALIDATION_SUMMARY.json", output)
    lines = [
        "# Synthèse des expériences de développement", "",
        (f"Meilleure macro-F1 de validation actuelle : **{results[best]['label']}**, "
         f"**{results[best]['validation']['macro_f1']:.4f}**. "
         "Ce classement sert au développement ; le jeu de test reste non évalué."), "",
        "## Résultats sur les mêmes morceaux", "",
        ("Toutes les expériences utilisent 6 299 morceaux de train et 780 de validation. "
         "La fenêtre centrale utilise un extrait par morceau ; les autres systèmes en utilisent trois. "
         "Les normalisations apprises sont ajustées uniquement sur train."), "",
        "| Système | Macro-F1 validation | Accuracy validation | Macro-F1 train | Sélection |",
        "|---|---:|---:|---:|---|",
    ]
    for name in ordered:
        item = results[name]
        if name in ("cnn", "cnn20"):
            chosen = ("graines 42, 43, 44" if repetition_summary is not None else
                      f"époque {item['selection']['epoch']}, graine {item['selection']['seed']}")
        else:
            chosen = f"C = {item['selection']['c']:g}"
        lines.append(f"| {item['label']} | {item['validation']['macro_f1']:.4f} | "
                     f"{item['validation']['accuracy']:.4f} | {item['training_macro_f1']:.4f} | {chosen} |")
    lines += [f"| Prédicteur majoritaire | {majority['macro_f1']:.4f} | {majority['accuracy']:.4f} | — | classe de train |"]
    if repetition_summary is not None:
        stats = repetition_summary["macro_f1"]
        lines += ["", (f"CNN : macro-F1 moyenne **{stats['mean']:.4f} ± {stats['sample_std']:.4f}** "
                        "(écart-type d’échantillon entre trois graines). Le tableau utilise la moyenne "
                        "des métriques des trois réseaux, pas un ensemble ni la meilleure graine. "
                        "Voir [les répétitions](REPETITIONS_CNN.md) pour chaque run et les limites.")]
    if extension_complete:
        stats = extension_summary["macro_f1"]
        lines += ["", (f"CNN, plafond vingt époques : **{stats['mean']:.4f} ± {stats['sample_std']:.4f}**, "
                        f"différence moyenne appariée {differences['macro_f1']['mean']:+.4f}. "
                        "Cette prolongation a été décidée après inspection des courbes initiales ; "
                        "elle augmente le nombre d’époques candidates sur validation. "
                        "Voir [l’extension CNN](EXTENSION_CNN.md).")]
    lines += [
              "", "## Ce que les comparaisons permettent de conclure", "",
              ("- Fusion : l’apport des descripteurs est étudié avec le bloc 12, ainsi que le retrait "
               "du rythme ou de l’harmonie. Aucune fusion avec le bloc 6 n’a été testée."),
              ("- Fenêtres : centrale seule contre trois fenêtres au bloc 12 ; cela modifie aussi "
               "le nombre d’exemples d’apprentissage, pas seulement l’inférence."),
              ("- Profondeur : bloc 6 contre bloc 12 à canaux identiques, avec la convention "
               "de LayerNorm finale gelée documentée."),
              ("- Canaux : original/H/P contre original répété, aux blocs 12 et 6. "
               "La combinaison bloc 6 + HPSS a été décidée après les ablations initiales ; "
               "elle complète les quatre configurations profondeur × canaux, avec C choisi "
               "séparément dans la même grille. Les gains ne sont pas supposés additifs."),
              ("- CNN contre ViT : architectures, préentraînements et modes d’apprentissage "
               "différents. Cette comparaison ne prouve pas un effet isolé de l’architecture."), "",
              ("Les métriques de validation ont été recalculées depuis les probabilités sauvegardées ; "
               "rapports, modèles et prédictions sont vérifiés par empreintes. Les mêmes identifiants "
               "et étiquettes de validation sont exigés. Les comparateurs propres à chaque ablation "
               "contrôlent en plus leurs caches et leurs paramètres."), "",
              "## Limites et prochaines étapes", "",
              ("C et les époques ont été sélectionnés sur validation. Les résultats du tableau sont "
               "donc des estimations de développement, sans intervalle de confiance final ni affirmation "
               "de significativité. Répéter les régressions "
               "déterministes ne constituerait pas une mesure de variabilité comparable."), "",
              ("Le budget initial des deux extractions supplémentaires d’ablation est consommé. "
               "Le plan reste évolutif : un point de décision sur les "
               "erreurs et les nouvelles hypothèses précède le verrouillage. Les extensions retenues "
               "auront un protocole et un budget explicites. Viendront ensuite les analyses de robustesse "
               "et l’évaluation du test avec bootstrap groupé par artiste. La démo utilisera "
               "le même prétraitement et le même artefact exporté."), "",
              "## Rapports et reproduction", "",
              "Commande : `.\\.venv\\Scripts\\python.exe scripts/summarize_validation.py`.", "",
              "- [Résumé machine et empreintes des expériences](VALIDATION_SUMMARY.json)",
              "- [ViT et descripteurs](COMPARAISON_VIT.md)",
              "- [Fusion](COMPARAISON_FUSION.md)",
              "- [CNN](BASELINE_CNN.md)",
              "- [Fenêtres](ABLATION_FENETRES.md)",
              "- [Blocs](ABLATION_BLOCS.md)",
              "- [Canaux HPSS](ABLATION_HPSS.md)",
              "- [Combinaison bloc 6 + HPSS](COMBINAISON_BLOC6_HPSS.md)",
              "- [Corrections et régressions de la combinaison](REGRESSIONS_BLOC6_HPSS.md)", ""]
    (ROOT / "reports/SYNTHESE_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"best": best, "macro_f1": results[best]["validation"]["macro_f1"]}, indent=2))


if __name__ == "__main__":
    main()
