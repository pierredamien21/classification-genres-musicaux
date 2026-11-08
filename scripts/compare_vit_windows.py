"""Validate the one-versus-three-window ablation and a fixed-model diagnostic."""

import csv
import json
from pathlib import Path

import numpy as np
from compare_vit_baseline import read_run

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.metrics import aggregate_tracks, classification_metrics
from music_genre.features.windows import central_window_rows
from music_genre.models.linear import LinearProbabilityModel

ROOT = Path(__file__).resolve().parents[1]


def main():
    tp, three, tf, tr, tc = read_run("vit")
    cp, central, cf, cr, cc = read_run("vit_central")
    for key in ("baseline_manifest_sha256", "labels", "track_counts", "feature_dimensions"):
        if three[key] != central[key]:
            raise ValueError(f"Incomparable window ablation: {key}")
    if ({k: v for k, v in three["config"].items() if k != "name"}
            != {k: v for k, v in central["config"].items() if k != "name"}):
        raise ValueError("Different classifier protocols")
    if any(central["window_counts"][s] != central["track_counts"][s]
           or three["window_counts"][s] != 3 * three["track_counts"][s]
           for s in ("train", "validation")):
        raise ValueError("Unexpected window counts")
    if [(r["track_id"], r["true_label"]) for r in cr] != [(r["track_id"], r["true_label"]) for r in tr]:
        raise ValueError("Different track predictions")
    if any(r["test_loaded"] or r["test_evaluated"] for r in (three, central)):
        raise ValueError("Window ablation must exclude test")
    parent = cf["provenance"]["parent"]
    if (parent["folder"] != three["feature_cache"]
            or parent["report_sha256"] != three["feature_report_sha256"]):
        raise ValueError("Central features belong to a different ViT reference")
    if ({k: v for k, v in tf["provenance"]["config"].items() if k not in ("name", "offsets_seconds")}
            != {k: v for k, v in cf["provenance"]["config"].items() if k not in ("name", "offsets_seconds")}):
        raise ValueError("Encoder/preprocessing differs")
    if (tf["provenance"]["config"]["offsets_seconds"] != [0., 10., 20.]
            or cf["provenance"]["config"]["offsets_seconds"] != [10.]):
        raise ValueError("Unexpected audio offsets")
    parent_cache, central_cache = ROOT / three["feature_cache"], ROOT / central["feature_cache"]
    with (parent_cache / "index.csv").open(encoding="utf-8", newline="") as stream:
        parent_index = list(csv.DictReader(stream))
    with (central_cache / "index.csv").open(encoding="utf-8", newline="") as stream:
        central_index = list(csv.DictReader(stream))
    selected, expected_index = central_window_rows(parent_index)
    x_parent = np.load(parent_cache / "features.npy", allow_pickle=False)
    x_central = np.load(central_cache / "features.npy", allow_pickle=False)
    if central_index != expected_index or not np.array_equal(x_central, x_parent[selected]):
        raise ValueError("Central features do not match the exact parent rows")
    # Diagnostic added after the primary result: frozen three-window classifier, central input only.
    positions = [i for i, r in enumerate(central_index) if r["split"] == "validation"]
    val_index = [central_index[i] for i in positions]
    model = LinearProbabilityModel.load(ROOT / tp["folder"] / "model.npz")
    ids, truth, proba = aggregate_tracks(val_index, model.predict_proba(x_central[positions]), len(three["labels"]))
    diagnostic = classification_metrics(truth, proba, three["labels"])
    diagnostic_rows = [
        {"track_id": key, "true_label": three["labels"][int(y)],
         **{f"p_{label}": float(p[i]) for i, label in enumerate(three["labels"])}}
        for key, y, p in zip(ids, truth, proba, strict=True)
    ]
    diagnostic_path = ROOT / cp["folder"] / "diagnostic_three_train_central_eval.csv"
    with diagnostic_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(diagnostic_rows[0]))
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    comparison = {
        "three_window_run": tp, "central_window_run": cp,
        "central_rows_equal_parent_subset": True, "tracks": three["track_counts"],
        "macro_f1_difference_three_minus_central": three["validation"]["macro_f1"] - central["validation"]["macro_f1"],
        "three_correct_central_wrong": int((tc & ~cc).sum()),
        "central_correct_three_wrong": int((~tc & cc).sum()),
        "diagnostic_three_train_central_eval": diagnostic,
        "diagnostic_predictions_sha256": sha256_file(diagnostic_path),
        "diagnostic_policy": "post_primary_result_fixed_model_no_refit_or_hyperparameter_selection",
        "test_evaluated": False,
    }
    write_json(ROOT / "reports/VIT_WINDOWS_COMPARISON.json", comparison)
    lines = [
        "# ViT : fenêtre centrale seule contre trois fenêtres", "",
        ("Même ViT gelé, même prétraitement, mêmes 6 299 morceaux de train et 780 de validation. "
         "La variante centrale utilise uniquement [10,20] secondes, à l’entraînement et à l’évaluation. "
         "La référence utilise [0,10], [10,20] et [20,30] secondes. Aucun morceau n’est supprimé."), "",
        "## Ablation principale prévue au plan", "",
        "| Fenêtres train → évaluation | Fenêtres de train | C retenu | Macro-F1 validation | Accuracy | Macro-F1 train |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, report in (("3 → 3", three), ("1 centrale → 1 centrale", central)):
        lines.append(f"| {name} | {report['window_counts']['train']} | {report['selected_c']:g} | "
                     f"{report['validation']['macro_f1']:.4f} | {report['validation']['accuracy']:.4f} | "
                     f"{report['training']['macro_f1']:.4f} |")
    lines += ["", f"Écart trois fenêtres − centrale : **{comparison['macro_f1_difference_three_minus_central']:+.4f}** en macro-F1.",
              "", "## Grille complète", "", "| C | Macro-F1 trois fenêtres | Macro-F1 centrale |",
              "|---:|---:|---:|"]
    for a, b in zip(three["trials"], central["trials"], strict=True):
        if a["c"] != b["c"]:
            raise ValueError("Unaligned C grid")
        lines.append(f"| {a['c']:g} | {a['validation']['macro_f1']:.4f} | {b['validation']['macro_f1']:.4f} |")
    lines += ["", "## Contrôle complémentaire à modèle fixe", "",
              ("Après le résultat principal, le classifieur déjà retenu sur trois fenêtres est réévalué "
               "sur les seules fenêtres centrales. Son scaler, ses coefficients et C sont inchangés. "
               "Ce diagnostic ne constitue pas un nouveau réglage ni un modèle retenu pour le test."), "",
              "| Modèle entraîné sur | Évaluation sur | Macro-F1 validation | Accuracy |",
              "|---|---|---:|---:|",
              f"| Trois fenêtres | Trois fenêtres | {three['validation']['macro_f1']:.4f} | {three['validation']['accuracy']:.4f} |",
              f"| Trois fenêtres | Centrale seule | {diagnostic['macro_f1']:.4f} | {diagnostic['accuracy']:.4f} |", "",
              ("Cette paire conserve le modèle et change uniquement les fenêtres moyennées à l’évaluation. "
               "L’ablation principale change aussi le nombre d’exemples et la diversité temporelle pendant "
               "l’entraînement : son écart ne mesure donc pas seulement un effet de moyenne des probabilités."), "",
              "## F1 par genre", "", "| Genre | Effectif | Trois fenêtres | Centrale seule |",
              "|---|---:|---:|---:|"]
    for label in three["labels"]:
        a, b = three["validation"]["per_class"][label], central["validation"]["per_class"][label]
        lines.append(f"| {label} | {int(a['support'])} | {a['f1-score']:.4f} | {b['f1-score']:.4f} |")
    lines += ["",
              (f"Trois fenêtres correctes, centrale incorrecte : {comparison['three_correct_central_wrong']} morceaux ; "
               f"centrale correcte, trois fenêtres incorrectes : {comparison['central_correct_three_wrong']}."), "",
              "## Calcul, vérifications et limites", "",
              (f"Entraînement/export central : {central['elapsed_seconds']:.2f} s. "
               f"Écart maximal après rechargement : {central['export_max_probability_error']:.3g}."),
              ("Les embeddings déjà calculés sont réutilisés sans redécodage ni nouvelle inférence ViT. "
               "À l’inférence, une fenêtre exige un passage encodeur au lieu de trois ; la durée totale "
               "n’est pas nécessairement divisée par trois, car décodage et chargement ont leur propre coût."), "",
              ("Index vérifié, une fenêtre centrale exactement par morceau, colonnes et vecteurs identiques "
               "au sous-ensemble du cache parent. Scaler appris sur les seules fenêtres de train retenues. "
               "Mêmes grille C et règles de sélection. Rapports, matrices, modèles et prédictions vérifiés "
               "par empreintes ; métriques recalculées depuis les prédictions sauvegardées."), "",
              ("Les scores restent des résultats de développement sur FMA-small filtré. La validation "
               "sélectionne C et sert aux comparaisons ; aucun intervalle ni écart significatif sur de "
               "nouveaux artistes n’est revendiqué. Le test reste non évalué."), "",
              "## Preuves et suite", "",
              "Voir [le protocole de l’ablation](../docs/ABLATION_FENETRES.md).", "",
              f"- [Rapport trois fenêtres](../{tp['folder']}/report.json)",
              f"- [Rapport centrale](../{cp['folder']}/report.json)",
              f"- [Prédictions du diagnostic](../{cp['folder']}/diagnostic_three_train_central_eval.csv)",
              "- [Comparaison vérifiable](VIT_WINDOWS_COMPARISON.json)", "",
              ("Prochain jalon : comparaison du bloc 6 et du bloc 12 du ViT, avec le même prétraitement, "
               "les trois fenêtres et la même grille. Benchmark préalable ; au maximum deux extractions "
               "supplémentaires pour les ablations, conformément au plan."), ""]
    (ROOT / "reports/ABLATION_FENETRES.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({k: v for k, v in comparison.items() if k != "diagnostic_three_train_central_eval"}, indent=2))
    print(f"Fixed-model central diagnostic: macro-F1={diagnostic['macro_f1']:.4f}, accuracy={diagnostic['accuracy']:.4f}")


if __name__ == "__main__":
    main()
