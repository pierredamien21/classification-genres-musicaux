"""Verify fusion ancestry and compare all prespecified validation experiments."""

import json
from pathlib import Path

import numpy as np
from compare_vit_baseline import read_run, validate_comparable

from music_genre.data.audit import write_json

ROOT = Path(__file__).resolve().parents[1]
NAMES = {
    "descriptor": "Descripteurs", "vit": "ViT seul", "fusion": "Fusion complète",
    "fusion_no_rhythm": "Fusion sans rythme", "fusion_no_harmony": "Fusion sans harmonie",
}


def main():
    runs = {key: read_run(key) for key in NAMES}
    _, vit, vf, vp, vc = runs["vit"]
    _, descriptor, df, _, _ = runs["descriptor"]
    vx = np.load(ROOT / vit["feature_cache"] / "features.npy", allow_pickle=False)
    dx = np.load(ROOT / descriptor["feature_cache"] / "features.npy", allow_pickle=False)
    results = {}
    for key, (pointer, report, features, predictions, correct) in runs.items():
        validate_comparable(vit, report, vf, features, vp, predictions)
        if key.startswith("fusion"):
            parents = features["provenance"]["parents"]
            for name, expected in (("vit", vit), ("descriptors", descriptor)):
                if (parents[name]["folder"] != expected["feature_cache"]
                        or parents[name]["report_sha256"] != expected["feature_report_sha256"]):
                    raise ValueError("Fusion was built from another reference representation")
            excluded = {"fusion": None, "fusion_no_rhythm": "rhythm",
                        "fusion_no_harmony": "harmony"}[key]
            columns = [i for i, family in enumerate(df["feature_families"]) if family != excluded]
            config = features["provenance"]["config"]
            if config["descriptor_columns"] != columns or config["excluded_family"] != excluded:
                raise ValueError("Incorrect ablation feature selection")
            matrix = np.load(ROOT / report["feature_cache"] / "features.npy", allow_pickle=False)
            if (not np.array_equal(matrix[:, :vx.shape[1]], vx)
                    or not np.array_equal(matrix[:, vx.shape[1]:], dx[:, columns])):
                raise ValueError("Fused values differ from their declared parents")
        results[key] = {
            "run": pointer, "dimensions": report["feature_dimensions"],
            "selected_c": report["selected_c"], "macro_f1": report["validation"]["macro_f1"],
            "accuracy": report["validation"]["accuracy"],
            "macro_f1_difference_from_vit": report["validation"]["macro_f1"] - vit["validation"]["macro_f1"],
            "corrected_vit_errors": int((correct & ~vc).sum()),
            "lost_vit_correct": int((~correct & vc).sum()),
        }
    write_json(ROOT / "reports/FUSION_COMPARISON.json", {
        "runs": results, "baseline_manifest_sha256": vit["baseline_manifest_sha256"],
        "index_sha256": vf["index_sha256"], "parent_values_verified": True,
        "test_evaluated": False,
    })
    lines = [
        "# Fusion ViT et descripteurs — résultats et ablations", "",
        ("Expériences sur les mêmes 6 299 morceaux d’entraînement et 780 morceaux de validation, "
         "avec 21 237 fenêtres alignées. Les deux ablations retirent séparément les familles "
         "rythme et harmonie prévues dans le plan. Aucun nouvel audio n’a été décodé."), "",
        "## Comparaison principale", "",
        "| Système | Dimensions | C choisi | Macro-F1 validation | Accuracy | Macro-F1 train | Écart au ViT |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for key, name in NAMES.items():
        report = runs[key][1]
        r = results[key]
        lines.append(f"| {name} | {r['dimensions']} | {r['selected_c']:g} | {r['macro_f1']:.4f} | "
                     f"{r['accuracy']:.4f} | {report['training']['macro_f1']:.4f} | "
                     f"{r['macro_f1_difference_from_vit']:+.4f} |")
    lines += ["", "## Grille complète et comparaison à C fixe", "",
              ("Chaque système dispose du même budget : trois valeurs de C, sélection sur la macro-F1 "
               "par morceau. La colonne C = 0,1 permet aussi de comparer les retraits de familles "
               "au même niveau de régularisation. Aucun essai n’est omis."), "",
              "| Système | Macro-F1 C = 0,1 | Macro-F1 C = 1 | Macro-F1 C = 10 |",
              "|---|---:|---:|---:|"]
    for key, name in NAMES.items():
        scores = [t["validation"]["macro_f1"] for t in runs[key][1]["trials"]]
        lines.append(f"| {name} | " + " | ".join(f"{score:.4f}" for score in scores) + " |")
    lines += ["", "## F1 par genre", "",
              "| Genre | Effectif | " + " | ".join(NAMES.values()) + " |",
              "|---|---:|" + "---:|" * len(NAMES)]
    for label in vit["labels"]:
        support = int(vit["validation"]["per_class"][label]["support"])
        scores = [runs[key][1]["validation"]["per_class"][label]["f1-score"] for key in NAMES]
        lines.append(f"| {label} | {support} | " + " | ".join(f"{s:.4f}" for s in scores) + " |")
    lines += ["", "## Erreurs par rapport au ViT", "",
              "| Système | Erreurs du ViT corrigées | Bonnes prédictions du ViT perdues |",
              "|---|---:|---:|"]
    for key in ("fusion", "fusion_no_rhythm", "fusion_no_harmony"):
        r = results[key]
        lines.append(f"| {NAMES[key]} | {r['corrected_vit_errors']} | {r['lost_vit_correct']} |")
    lines += ["", "## Coût et traçabilité", "",
              "| Expérience | Assemblage cache (s) | Entraînement et export (s) | Écart export maximal |",
              "|---|---:|---:|---:|"]
    for key in ("fusion", "fusion_no_rhythm", "fusion_no_harmony"):
        report, features = runs[key][1:3]
        lines.append(f"| {NAMES[key]} | {features['elapsed_seconds']:.3f} | {report['elapsed_seconds']:.3f} | "
                     f"{report['export_max_probability_error']:.3g} |")
    lines += ["",
              ("Les temps d’assemblage réutilisent les caches existants ; ils n’incluent pas les "
               "9,49 minutes d’extraction ViT ni les 11,68 minutes de descripteurs. À l’inférence "
               "sur un nouveau morceau, une fusion doit calculer les deux représentations."), "",
              ("Vérifications : empreintes des caches, des modèles et des prédictions ; index identiques ; "
               "valeurs fusionnées égales à leurs colonnes sources ; même grille ; métriques recalculées "
               "depuis les prédictions exportées. Standardisation ajustée sur train, neuf ajustements "
               "de fusion convergés et exports numériques rechargés."), "",
              "## Interprétation", "",
              ("La concaténation complète n’améliore pas le ViT seul dans ce protocole. Retirer le rythme "
               "ou l’harmonie relève son score, mais aucune de ces variantes ne dépasse le ViT. "
               "L’hypothèse d’un gain par ajout des descripteurs n’est donc pas soutenue par ces essais."), "",
              ("Ce constat ne signifie pas que rythme ou harmonie seraient inutiles pour reconnaître "
               "les genres. Il concerne ces résumés statistiques, cette représentation ViT, ce classifieur "
               "et cette sélection de données. Redondance, interactions, régularisation et estimation "
               "sur un échantillon fini sont des explications possibles, pas des causes démontrées."), "",
              ("La standardisation par dimension ne donne pas le même poids global à chaque famille : "
               "elles contiennent des nombres de dimensions différents. La concaténation augmente aussi "
               "la capacité du classifieur. Les ablations mesurent un retrait de famille dans ce système, "
               "pas une importance musicale universelle."), "",
              ("La validation a servi à choisir C et à comparer plusieurs systèmes. Les écarts observés "
               "ne prouvent pas une différence significative sur de nouveaux artistes. Le test reste "
               "réservé au protocole final ; aucune nouvelle recherche de paramètres n’est ajoutée "
               "à cette étape pour renverser le résultat."), "",
              "## Suite et reproduction", "",
              ("Conserver le ViT seul comme meilleure référence actuelle selon la macro-F1 de validation. "
               "Prochain jalon : benchmark du petit CNN prévu au plan, puis entraînement dans le budget "
               "CPU si sa faisabilité est confirmée. Le choix final sera verrouillé avant le test."), "",
              "Voir [le protocole de fusion](../docs/FUSION.md) et [les résultats vérifiables](FUSION_COMPARISON.json).", ""]
    for key, name in NAMES.items():
        lines.append(f"- [{name} : rapport d’expérience](../{runs[key][0]['folder']}/report.json)")
    lines.append("")
    (ROOT / "reports/COMPARAISON_FUSION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
