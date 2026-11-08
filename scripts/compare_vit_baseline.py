"""Verify comparable validation runs and write a reproducible French comparison."""

import csv
import json
from pathlib import Path

import numpy as np

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.metrics import classification_metrics

ROOT = Path(__file__).resolve().parents[1]


def read_run(name):
    pointer = json.loads((ROOT / f"reports/latest_{name}_run.json").read_text())
    folder = (ROOT / pointer["folder"]).resolve()
    if not folder.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Run outside workspace runs")
    if sha256_file(folder / "report.json") != pointer["report_sha256"]:
        raise ValueError("Run report changed")
    report = json.loads((folder / "report.json").read_text())
    for filename, key in (("model.npz", "model_sha256"),
                          ("validation_predictions.csv", "validation_predictions_sha256")):
        if sha256_file(folder / filename) != report[key]:
            raise ValueError(f"Run artifact changed: {filename}")
    cache = (ROOT / report["feature_cache"]).resolve()
    if not cache.is_relative_to((ROOT / "data/features").resolve()):
        raise ValueError("Feature cache outside workspace")
    if sha256_file(cache / "report.json") != report["feature_report_sha256"]:
        raise ValueError("Feature report changed")
    features = json.loads((cache / "report.json").read_text())
    for filename, key in (("index.csv", "index_sha256"), ("features.npy", "features_sha256")):
        if sha256_file(cache / filename) != features[key]:
            raise ValueError(f"Feature artifact changed: {filename}")
    with (folder / "validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
        predictions = list(csv.DictReader(stream))
    labels = report["labels"]
    truth = np.array([labels.index(row["true_label"]) for row in predictions])
    probabilities = np.array([[float(row[f"p_{label}"]) for label in labels] for row in predictions])
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
        raise ValueError("Invalid probabilities")
    metrics = classification_metrics(truth, probabilities, labels)
    if metrics != report["validation"]:
        raise ValueError("Recomputed validation metrics differ")
    correct = probabilities.argmax(axis=1) == truth
    return pointer, report, features, predictions, correct


def validate_comparable(d, v, df, vf, dr, vr):
    for key in ("baseline_manifest_sha256", "labels", "window_counts", "track_counts"):
        if d[key] != v[key]:
            raise ValueError(f"Incomparable runs: {key}")
    if df["index_sha256"] != vf["index_sha256"]:
        raise ValueError("Representations use different windows")
    if {k: value for k, value in d["config"].items() if k != "name"} != {
        k: value for k, value in v["config"].items() if k != "name"
    }:
        raise ValueError("Classifier protocols differ")
    if [(r["track_id"], r["true_label"]) for r in dr] != [
        (r["track_id"], r["true_label"]) for r in vr
    ]:
        raise ValueError("Predictions are not aligned")
    if any(r["test_loaded"] or r["test_evaluated"] for r in (d, v)):
        raise ValueError("This comparison is restricted to validation")


def main():
    dp, d, df, dr, dc = read_run("descriptor")
    vp, v, vf, vr, vc = read_run("vit")
    validate_comparable(d, v, df, vf, dr, vr)
    overlap = {
        "both_correct": int((dc & vc).sum()),
        "only_descriptors_correct": int((dc & ~vc).sum()),
        "only_vit_correct": int((~dc & vc).sum()),
        "both_wrong": int((~dc & ~vc).sum()),
    }
    comparison = {
        "descriptor_run": dp, "vit_run": vp,
        "identical_window_index_sha256": vf["index_sha256"],
        "baseline_manifest_sha256": v["baseline_manifest_sha256"],
        "validation_tracks": v["track_counts"]["validation"],
        "macro_f1_difference_vit_minus_descriptors": v["validation"]["macro_f1"] - d["validation"]["macro_f1"],
        "paired_correctness": overlap, "test_evaluated": False,
    }
    write_json(ROOT / "reports/VIT_COMPARISON.json", comparison)
    lines = [
        "# ViT gelé et descripteurs musicaux — comparaison sur validation", "",
        ("Même sélection filtrée de FMA-small : 6 299 morceaux d’entraînement, 780 de validation ; "
         "21 237 fenêtres identiques. Le jeu de test n’est ni extrait ni évalué pour cette expérience."), "",
        "## Résultats par morceau", "",
        "| Système | Dimensions | C choisi | Macro-F1 validation | Accuracy validation | Macro-F1 train |",
        "|---|---:|---:|---:|---:|---:|",
        f"| Majoritaire ({d['majority_class']}) | — | — | {d['majority_validation']['macro_f1']:.4f} | {d['majority_validation']['accuracy']:.4f} | — |",
    ]
    for name, run in (("Descripteurs", d), ("ViT gelé", v)):
        lines.append(f"| {name} | {run['feature_dimensions']} | {run['selected_c']:g} | "
                     f"{run['validation']['macro_f1']:.4f} | {run['validation']['accuracy']:.4f} | "
                     f"{run['training']['macro_f1']:.4f} |")
    lines += ["", f"Écart ViT − descripteurs : **{comparison['macro_f1_difference_vit_minus_descriptors']:+.4f}** en macro-F1.",
              "", "## Grille ViT", "", "| C | Macro-F1 validation | Accuracy | Itérations |",
              "|---:|---:|---:|---:|"]
    for trial in v["trials"]:
        lines.append(f"| {trial['c']:g} | {trial['validation']['macro_f1']:.4f} | "
                     f"{trial['validation']['accuracy']:.4f} | {trial['iterations'][0]} |")
    lines += ["", "## F1 par genre", "", "| Genre | Effectif validation | Descripteurs | ViT |",
              "|---|---:|---:|---:|"]
    for label in v["labels"]:
        dm, vm = d["validation"]["per_class"][label], v["validation"]["per_class"][label]
        lines.append(f"| {label} | {int(vm['support'])} | {dm['f1-score']:.4f} | {vm['f1-score']:.4f} |")
    lines += ["", "## Erreurs appariées", "",
              f"- Les deux systèmes corrects : {overlap['both_correct']} morceaux.",
              f"- Descripteurs seuls corrects : {overlap['only_descriptors_correct']}.",
              f"- ViT seul correct : {overlap['only_vit_correct']}.",
              f"- Les deux systèmes incorrects : {overlap['both_wrong']}.", "",
              ("Ces désaccords motivent l’expérience de fusion. Ils ne garantissent pas qu’un "
               "classifieur saura combiner les représentations et ne constituent pas un test de significativité."),
              "", "## Confusion ViT", "", "Lignes : genre réel ; colonnes : genre prédit.", "",
              "| Réel / prédit | " + " | ".join(v["labels"]) + " |",
              "|---|" + "---:|" * len(v["labels"])]
    for label, row in zip(v["labels"], v["validation"]["confusion_matrix"], strict=True):
        lines.append("| " + label + " | " + " | ".join(str(n) for n in row) + " |")
    lines += ["", "## Coût et contrôles", "",
              f"Extraction descripteurs : {df['elapsed_seconds'] / 60:.2f} min ; ViT : {vf['elapsed_seconds'] / 60:.2f} min.",
              ("Durées observées sur ce poste, quatre processus CPU ; elles dépendent de la charge "
               "et ne sont pas des mesures contrôlées de vitesse relative."),
              f"Entraînement et export ViT : {v['elapsed_seconds']:.2f} s. Écart maximal après rechargement : {v['export_max_probability_error']:.3g}.",
              ("Empreintes vérifiées : rapports, matrices, index, modèles et prédictions. Index de fenêtres "
               "identiques octet pour octet ; même grille C et même protocole. Métriques recalculées "
               "depuis les probabilités exportées. Standardisation ajustée sur train uniquement."),
              "", "## Interprétation et limites", "",
              ("Le ViT est préentraîné sur des images et reste intégralement gelé. Seule la tête logistique "
               "est entraînée sur FMA. Le redimensionnement carré compresse le temps ; la plage dB fixe "
               "n’a pas été optimisée. Cette expérience mesure une référence de transfert, pas la "
               "performance maximale d’un Transformer audio."), "",
              ("Le choix de C utilise la validation : ces scores servent au développement et ne sont pas "
               "une estimation finale indépendante. Aucun intervalle de confiance ni gain sur test n’est revendiqué. "
               "Les catégories sont les huit genres FMA ; le gospel n’est pas une classe de cette version."),
              "", "## Reproduction et preuves", "",
              "Voir [le protocole ViT](../docs/VIT_GELE.md).", "",
              f"- [Rapport descripteurs](../{dp['folder']}/report.json)",
              f"- [Rapport ViT](../{vp['folder']}/report.json)",
              "- [Comparaison vérifiable](VIT_COMPARISON.json)",
              "- [Benchmark de 64 fenêtres](VIT_BENCHMARK.json)", ""]
    (ROOT / "reports/COMPARAISON_VIT.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
