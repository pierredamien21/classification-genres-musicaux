"""Create a readable validation report from the verified first descriptor experiment."""

import csv
import json
from pathlib import Path

from music_genre.data.audit import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    pointer = json.loads((ROOT / "reports/latest_descriptor_run.json").read_text())
    folder = (ROOT / pointer["folder"]).resolve()
    if not folder.is_relative_to((ROOT / "runs").resolve()):
        raise ValueError("Experiment outside runs")
    if sha256_file(folder / "report.json") != pointer["report_sha256"]:
        raise ValueError("Experiment report changed")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    for filename, field in (("model.npz", "model_sha256"),
                            ("validation_predictions.csv", "validation_predictions_sha256")):
        if sha256_file(folder / filename) != report[field]:
            raise ValueError(f"Experiment artifact changed: {filename}")
    cache = (ROOT / report["feature_cache"]).resolve()
    if not cache.is_relative_to((ROOT / "data/features").resolve()):
        raise ValueError("Cache outside features")
    if sha256_file(cache / "report.json") != report["feature_report_sha256"]:
        raise ValueError("Feature report changed")
    features = json.loads((cache / "report.json").read_text(encoding="utf-8"))
    prefix = "../" + folder.relative_to(ROOT).as_posix()
    val, train, majority = report["validation"], report["training"], report["majority_validation"]
    lines = [
        "# Première référence : descripteurs musicaux", "",
        (f"Expérience du {report['created_at_utc'][:10]} sur le manifeste filtré "
        f"`{report['baseline_manifest_sha256']}`."), "",
        "**Résultats de validation uniquement : le jeu de test n’a pas été évalué.**", "",
        "| Système | Macro-F1 par morceau | Accuracy |", "|---|---:|---:|",
        (f"| Genre majoritaire ({report['majority_class']}) | {majority['macro_f1']:.4f} | "
        f"{majority['accuracy']:.4f} |"),
        (f"| Descripteurs + régression logistique | **{val['macro_f1']:.4f}** | "
        f"**{val['accuracy']:.4f}** |"), "",
        (f"Évaluation sur {report['track_counts']['validation']} morceaux de validation. "
        f"Entraînement sur {report['track_counts']['train']} morceaux "
        f"({report['window_counts']['train']} fenêtres). "
        "Les probabilités des fenêtres sont moyennées par morceau avant les métriques."),
        "", "## Ce que représente chaque fenêtre", "",
        "| Famille | Dimensions | Informations résumées |", "|---|---:|---|",
        ("| Timbre et énergie | 60 | 20 MFCC, centroïde, largeur et contraste spectraux, RMS ; "
        "moyennes et écarts-types |"),
        "| Harmonie | 24 | Moyennes et écarts-types du chroma à 12 classes de hauteur |",
        "| Rythme | 41 | Autocorrélation du flux spectral sur une grille 40–240 BPM |",
        "", ("Mono 16 kHz, fenêtres à 0, 10 et 20 secondes, FFT 1 024, pas 256. Les statistiques "
        "utilisent les trames complètes du signal réel ; une seule trame complétée est autorisée "
        "si le signal est plus court que la FFT. Le padding final n’ajoute donc pas de faux silence "
        "aux moyennes. Les paramètres exacts sont conservés dans l’expérience."),
        "", ("Le chroma n’est pas une transcription d’accords ; le profil rythmique peut contenir "
        "plusieurs périodicités et ne constitue pas une mesure certaine du tempo."),
        "", "## Sélection du classifieur", "",
        ("StandardScaler est ajusté uniquement sur les fenêtres de training. Même famille "
        "de classifieur pour les trois valeurs de C prévues ; sélection par macro-F1 de validation "
        "et préférence au plus petit C en cas d’égalité."), "",
        "| C | Macro-F1 validation | Accuracy validation | Itérations |",
        "|---:|---:|---:|---|",
    ]
    for trial in report["trials"]:
        lines.append(f"| {trial['c']} | {trial['validation']['macro_f1']:.4f} | "
                     f"{trial['validation']['accuracy']:.4f} | {trial['iterations']} |")
    lines += [
        "", (f"**C retenu : {report['selected_c']}.** Les modèles ont convergé. "
        f"Macro-F1 d’entraînement du modèle retenu : {train['macro_f1']:.4f}. "
        "L’écart avec la validation est descriptif ; il ne remplace pas l’évaluation finale."),
        "", "## Résultats par genre sur validation", "",
        "| Genre | Précision | Rappel | F1 | Morceaux |", "|---|---:|---:|---:|---:|",
    ]
    for label in report["labels"]:
        item = val["per_class"][label]
        lines.append(f"| {label} | {item['precision']:.3f} | {item['recall']:.3f} | "
                     f"{item['f1-score']:.3f} | {int(item['support'])} |")
    lines += [
        "", "### Matrice de confusion", "",
        "Lignes : genre réel ; colonnes : genre prédit. Comptages par morceau.", "",
        "| Réel / Prédit | " + " | ".join(report["labels"]) + " |",
        "|---|" + "---:|" * len(report["labels"]),
    ]
    for label, counts in zip(report["labels"], val["confusion_matrix"], strict=True):
        lines.append("| " + label + " | " + " | ".join(map(str, counts)) + " |")
    with (folder / "validation_confusion.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["true_genre", *report["labels"]])
        for label, counts in zip(report["labels"], val["confusion_matrix"], strict=True):
            writer.writerow([label, *counts])
    lines += [
        "", "## Calcul, export et reproduction", "",
        (f"- Caractéristiques : {features['matrix_shape'][0]:,} fenêtres × "
        f"{features['matrix_shape'][1]} valeurs, toutes finies ; extraction en "
        f"{features['elapsed_seconds'] / 60:.1f} minutes sur CPU."),
        f"- Entraînement, sélection et export : {report['elapsed_seconds']:.1f} secondes.",
        (f"- Export numérique rechargé : différence maximale de probabilité "
        f"{report['export_max_probability_error']:.2e} par rapport à scikit-learn."),
        f"- [Provenance et résultats complets]({prefix}/report.json)",
        f"- [Prédictions de validation]({prefix}/validation_predictions.csv)",
        f"- [Configuration du modèle]({prefix}/model.json)",
        f"- [Export numérique sans pickle]({prefix}/model.npz)",
        f"- [Statistiques des caractéristiques sur train]({prefix}/train_feature_statistics.json)",
        "", "## Limites et suite", "",
        ("Le meilleur C est choisi sur cette validation : son score ne constitue donc pas une "
        "estimation finale indépendante. Aucun intervalle de confiance sur test n’est calculé. "
        "Ce sous-ensemble filtré de FMA-small doit être explicitement distingué du benchmark "
        "non filtré. Aucun gain de la fusion, du CNN ou du ViT n’est encore démontré."),
        "", ("Prochaine comparaison : extraire le ViT gelé sur les mêmes fenêtres et les mêmes "
        "morceaux, puis appliquer la même grille de classifieur. La fusion et les ablations "
        "suivront ; le test restera réservé au protocole final."), "",
    ]
    (ROOT / "reports/BASELINE_DESCRIPTEURS.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {ROOT / 'reports/BASELINE_DESCRIPTEURS.md'}")


if __name__ == "__main__":
    main()
