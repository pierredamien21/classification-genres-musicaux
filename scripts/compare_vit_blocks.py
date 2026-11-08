"""Compare block-six and block-twelve linear probes with unchanged evaluation protocol."""

import json
from pathlib import Path

from compare_vit_baseline import read_run, validate_comparable

from music_genre.data.audit import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    p12, r12, f12, rows12, correct12 = read_run("vit")
    p6, r6, f6, rows6, correct6 = read_run("vit_block6")
    validate_comparable(r12, r6, f12, f6, rows12, rows6)
    if r12["feature_dimensions"] != 192 or r6["feature_dimensions"] != 192:
        raise ValueError("Different representation dimensions")
    cfg12, cfg6 = f12["provenance"]["config"], f6["provenance"]["config"]
    allowed = {"name", "feature_block", "terminal_norm", "pooling"}
    if ({k: v for k, v in cfg12.items() if k not in allowed}
            != {k: v for k, v in cfg6.items() if k not in allowed}):
        raise ValueError("Unintended representation configuration difference")
    if (cfg6["feature_block"] != 6
            or cfg6["terminal_norm"] != "same_frozen_checkpoint_layernorm_after_selected_block"
            or cfg6["pooling"] != "selected_block_mean_patch_tokens_exclude_prefix"
            or cfg12["pooling"] != "last_block_mean_patch_tokens_exclude_prefix"):
        raise ValueError("Unexpected block extraction protocol")
    parent = f6["provenance"]["parents"]["vit_reference"]
    if parent != {"folder": r12["feature_cache"], "report_sha256": r12["feature_report_sha256"]}:
        raise ValueError("Block-six extraction references another block-twelve cache")
    benchmark_path = ROOT / "reports/VIT_BLOCK6_BENCHMARK.json"
    if sha256_file(benchmark_path) != f6["benchmark_sha256"]:
        raise ValueError("Benchmark changed since extraction")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if benchmark["cache_key"] != f6["cache_key"] or benchmark["sample_split"] != "train":
        raise ValueError("Incompatible benchmark")
    comparison = {
        "block12_run": p12, "block6_run": p6, "identical_window_index_sha256": f6["index_sha256"],
        "macro_f1_difference_block6_minus_block12": r6["validation"]["macro_f1"] - r12["validation"]["macro_f1"],
        "block6_correct_block12_wrong": int((correct6 & ~correct12).sum()),
        "block12_correct_block6_wrong": int((correct12 & ~correct6).sum()),
        "block12_reference_max_abs_error": benchmark["block12_reference_max_abs_error"],
        "test_evaluated": False,
    }
    write_json(ROOT / "reports/VIT_BLOCKS_COMPARISON.json", comparison)
    winner = "bloc 6" if comparison["macro_f1_difference_block6_minus_block12"] > 0 else "bloc 12"
    lines = [
        "# ViT : bloc 6 contre bloc 12", "",
        ("Ablation sur les mêmes 6 299 morceaux de train et 780 de validation, trois fenêtres par morceau, "
         "192 caractéristiques et grille C identique. Les poids du ViT restent gelés."), "",
        "## Résultats par morceau", "",
        "| Représentation | C retenu | Macro-F1 validation | Accuracy validation | Macro-F1 train |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, report in (("Bloc 12 (référence)", r12), ("Bloc 6", r6)):
        lines.append(f"| {name} | {report['selected_c']:g} | {report['validation']['macro_f1']:.4f} | "
                     f"{report['validation']['accuracy']:.4f} | {report['training']['macro_f1']:.4f} |")
    lines += ["", f"Écart bloc 6 − bloc 12 : **{comparison['macro_f1_difference_block6_minus_block12']:+.4f}** en macro-F1.",
              f"Le **{winner}** est retenu comme référence entre ces deux variantes selon le critère principal de validation.",
              "", "## Grille complète", "", "| C | Macro-F1 bloc 12 | Macro-F1 bloc 6 |", "|---:|---:|---:|"]
    for t12, t6 in zip(r12["trials"], r6["trials"], strict=True):
        if t12["c"] != t6["c"]:
            raise ValueError("Misaligned C grid")
        lines.append(f"| {t12['c']:g} | {t12['validation']['macro_f1']:.4f} | {t6['validation']['macro_f1']:.4f} |")
    lines += ["", "## F1 par genre", "", "| Genre | Effectif | Bloc 12 | Bloc 6 |", "|---|---:|---:|---:|"]
    for label in r12["labels"]:
        a, b = r12["validation"]["per_class"][label], r6["validation"]["per_class"][label]
        lines.append(f"| {label} | {int(a['support'])} | {a['f1-score']:.4f} | {b['f1-score']:.4f} |")
    lines += ["", "## Erreurs appariées", "",
              f"- Bloc 6 correct, bloc 12 incorrect : {comparison['block6_correct_block12_wrong']} morceaux.",
              f"- Bloc 12 correct, bloc 6 incorrect : {comparison['block12_correct_block6_wrong']} morceaux.",
              "", "## Confusion du bloc 6", "", "Lignes : genre réel ; colonnes : genre prédit.", "",
              "| Réel / prédit | " + " | ".join(r6["labels"]) + " |", "|---|" + "---:|" * len(r6["labels"])]
    for label, row in zip(r6["labels"], r6["validation"]["confusion_matrix"], strict=True):
        lines.append("| " + label + " | " + " | ".join(str(n) for n in row) + " |")
    lines += ["", "## Protocole et vérifications", "",
              ("Le même calcul de patches, les mêmes positions, six premiers blocs identiques et la même "
               "LayerNorm finale gelée du checkpoint sont utilisés. La variante s’arrête après le bloc 6 "
               "avant cette LayerNorm ; la référence traverse les douze blocs. Dans les deux cas, les "
               "tokens de préfixe sont exclus avant la moyenne des patches. Aucun poids n’est réajusté."), "",
              ("La LayerNorm finale a été apprise pour la sortie du bloc 12 pendant le préentraînement. "
               "Son emploi au bloc 6 est une convention explicite de cette ablation, pas une normalisation "
               "spécifiquement apprise pour la couche intermédiaire."), "",
              (f"Sur {benchmark['sample_count']} fenêtres de train, le chemin log-Mel en cache reproduit "
               f"les vecteurs du bloc 12 avec un écart maximal de {benchmark['block12_reference_max_abs_error']:.3g}. "
               "Le test unitaire compare aussi la sortie tronquée à une capture directe après le sixième bloc "
               "du réseau complet, suivie de la même LayerNorm."), "",
              ("Rapports, matrices, modèles, prédictions et benchmark contrôlés par empreintes. Index identiques "
               "octet pour octet ; mêmes labels, partitions et budgets de recherche. Métriques recalculées "
               "depuis les prédictions sauvegardées. Scalers ajustés uniquement sur les fenêtres de train."),
              "", "## Coût observé", "",
              f"- Médiane bloc 6 : {benchmark['median_block6_seconds']:.4f} s par fenêtre, après chauffe, un fil CPU.",
              f"- Extraction complète : {f6['elapsed_seconds'] / 60:.2f} min, {f6['workers']} processus.",
              f"- Grille logistique et export : {r6['elapsed_seconds']:.2f} s.",
              f"- Écart maximal après rechargement du classifieur : {r6['export_max_probability_error']:.3g}.", "",
              ("L’extraction réutilise les log-Mel déjà calculés pour le CNN. Sa durée ne peut pas être "
               "comparée directement à celle du bloc 12, qui incluait le décodage des MP3 et le calcul "
               "des spectrogrammes. Une des deux extractions supplémentaires prévues pour les ablations "
               "est consommée ; l’ablation centrale n’en a consommé aucune."), "",
              "## Limites et suite", "",
              ("Les résultats concernent ce checkpoint, ce pooling, cette convention de normalisation "
               "et le sous-ensemble filtré FMA-small. Ils ne déterminent pas la meilleure couche pour "
               "tout modèle audio. La validation choisit C et sert aux comparaisons : aucun score final "
               "indépendant ni écart significatif sur de nouveaux artistes n’est revendiqué. Le test reste non évalué."), "",
              ("Prochaine ablation prévue : canaux log-Mel original/harmonique/percussif contre répétition "
               "du log-Mel, sur la référence bloc 12 du plan pour ne changer qu’un facteur. Benchmark "
               "préalable et dernière extraction supplémentaire autorisée par le plan."), "",
              "## Reproduction et preuves", "", "Voir [le protocole](../docs/ABLATION_BLOCS.md).", "",
              f"- [Rapport bloc 12](../{p12['folder']}/report.json)",
              f"- [Rapport bloc 6](../{p6['folder']}/report.json)",
              "- [Comparaison vérifiable](VIT_BLOCKS_COMPARISON.json)",
              "- [Benchmark](VIT_BLOCK6_BENCHMARK.json)", ""]
    (ROOT / "reports/ABLATION_BLOCS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
