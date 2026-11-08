"""Compare channel representations while retaining the frozen block-12 reference."""

import json
from itertools import pairwise
from pathlib import Path

from compare_vit_baseline import read_run, validate_comparable

from music_genre.data.audit import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    p0, r0, f0, rows0, correct0 = read_run("vit")
    ph, rh, fh, rowsh, correcth = read_run("vit_hpss")
    validate_comparable(r0, rh, f0, fh, rows0, rowsh)
    c0, ch = f0["provenance"]["config"], fh["provenance"]["config"]
    allowed = {"name", "channels", "hpss"}
    if ({k: v for k, v in c0.items() if k not in allowed}
            != {k: v for k, v in ch.items() if k not in allowed}
            or ch["channels"] != "original_harmonic_percussive"
            or c0["channels"] != "repeat_grayscale_three_times"
            or ch["pooling"] != "last_block_mean_patch_tokens_exclude_prefix"
            or rh["feature_dimensions"] != 192 or r0["feature_dimensions"] != 192):
        raise ValueError("Unintended change to the representation protocol")
    if fh["provenance"]["parents"]["vit_reference"] != {
        "folder": r0["feature_cache"], "report_sha256": r0["feature_report_sha256"]
    }:
        raise ValueError("HPSS extraction references another baseline")
    benchmark_path = ROOT / "reports/VIT_HPSS_BENCHMARK.json"
    if sha256_file(benchmark_path) != fh["benchmark_sha256"]:
        raise ValueError("Benchmark changed since extraction")
    benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
    if (benchmark["cache_key"] != fh["cache_key"] or benchmark["split"] != "train"
            or benchmark["sample_count"] != 64
            or benchmark["original_logmel_max_abs_error"] > 1e-5
            or benchmark["repeated_original_embedding_max_abs_error"] > 1e-6):
        raise ValueError("Incompatible or failed benchmark")
    p6, r6, f6, rows6, _ = read_run("vit_block6")
    validate_comparable(r0, r6, f0, f6, rows0, rows6)
    cache_times = sorted(p.stat().st_mtime for p in
                         (ROOT / rh["feature_cache"] / "tracks").glob("*.json"))
    gaps = [b - a for a, b in pairwise(cache_times)]
    timing = {"track_files": len(cache_times), "largest_cache_write_gap_seconds": max(gaps, default=0),
              "gaps_over_two_minutes_seconds": [gap for gap in gaps if gap > 120]}
    delta = rh["validation"]["macro_f1"] - r0["validation"]["macro_f1"]
    comparison = {
        "original_run": p0, "hpss_run": ph, "block6_context_run": p6,
        "identical_window_index_sha256": fh["index_sha256"],
        "baseline_manifest_sha256": rh["baseline_manifest_sha256"],
        "macro_f1_difference_hpss_minus_original": delta,
        "hpss_correct_original_wrong": int((correcth & ~correct0).sum()),
        "original_correct_hpss_wrong": int((correct0 & ~correcth).sum()),
        "cache_write_timing": timing,
        "test_evaluated": False,
    }
    write_json(ROOT / "reports/VIT_HPSS_COMPARISON.json", comparison)
    lines = [
        "# ViT : original/harmonique/percussif contre log-Mel répété", "",
        ("Ablation au bloc 12 gelé, sur 6 299 morceaux de train et 780 de validation. "
         "Mêmes trois fenêtres, checkpoint, 192 caractéristiques et grille logistique."), "",
        "## Résultats par morceau", "",
        "| Canaux / profondeur | C | Macro-F1 validation | Accuracy | Macro-F1 train |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, report in (("Original répété / bloc 12", r0), ("Original + H + P / bloc 12", rh),
                           ("Original répété / bloc 6 (contexte)", r6)):
        lines.append(f"| {label} | {report['selected_c']:g} | {report['validation']['macro_f1']:.4f} | "
                     f"{report['validation']['accuracy']:.4f} | {report['training']['macro_f1']:.4f} |")
    lines += ["", f"Écart HPSS − original, à profondeur identique : **{delta:+.4f}** en macro-F1.",
              ("Cette variante HPSS améliore la référence bloc 12 sur validation." if delta > 0 else
               "Cette variante HPSS n’améliore pas la référence bloc 12 sur validation."), "",
              ("Le bloc 6 est affiché pour situer les résultats du projet. Sa comparaison avec HPSS "
               "change à la fois profondeur et canaux : elle ne mesure pas l’effet isolé de HPSS."),
              "", "## Grille complète", "", "| C | Original bloc 12 | HPSS bloc 12 |", "|---:|---:|---:|"]
    for a, b in zip(r0["trials"], rh["trials"], strict=True):
        if a["c"] != b["c"]:
            raise ValueError("Misaligned C grid")
        lines.append(f"| {a['c']:g} | {a['validation']['macro_f1']:.4f} | {b['validation']['macro_f1']:.4f} |")
    lines += ["", "## F1 par genre", "", "| Genre | Effectif | Original bloc 12 | HPSS bloc 12 |",
              "|---|---:|---:|---:|"]
    for label in r0["labels"]:
        a, b = r0["validation"]["per_class"][label], rh["validation"]["per_class"][label]
        lines.append(f"| {label} | {int(a['support'])} | {a['f1-score']:.4f} | {b['f1-score']:.4f} |")
    lines += ["", "## Erreurs appariées et confusion", "",
              f"- HPSS correct, original incorrect : {comparison['hpss_correct_original_wrong']} morceaux.",
              f"- Original correct, HPSS incorrect : {comparison['original_correct_hpss_wrong']} morceaux.", "",
              "Confusion HPSS : lignes réelles, colonnes prédites.", "",
              "| Réel / prédit | " + " | ".join(rh["labels"]) + " |", "|---|" + "---:|" * len(rh["labels"])]
    for label, row in zip(rh["labels"], rh["validation"]["confusion_matrix"], strict=True):
        lines.append("| " + label + " | " + " | ".join(str(n) for n in row) + " |")
    lines += ["", "## Vérifications et coût", "",
              ("Empreintes des rapports, matrices, modèles et prédictions vérifiées ; index identiques "
               "octet pour octet ; métriques recalculées depuis les probabilités exportées. Le scaler "
               "est appris exclusivement sur train et les trois ajustements logistiques ont convergé."), "",
              f"- Écart maximal canal original / ancien log-Mel : {benchmark['original_logmel_max_abs_error']:.3g} dB.",
              (f"- Écart maximal vecteur original répété / ancien cache, sur 64 fenêtres de train : "
               f"{benchmark['repeated_original_embedding_max_abs_error']:.3g}."),
              f"- Médiane HPSS + log-Mel + ViT : {benchmark['median_features_seconds']:.4f} s par fenêtre (un fil).",
              f"- Médiane décodage : {benchmark['median_decode_seconds']:.4f} s par morceau de benchmark.",
              f"- Durée écoulée enregistrée pour l’extraction : {fh['elapsed_seconds'] / 60:.2f} min, {fh['workers']} processus.",
              f"- Grille logistique et export : {rh['elapsed_seconds']:.2f} s.",
              f"- Écart de probabilités après rechargement du classifieur : {rh['export_max_probability_error']:.3g}.", "",
              ("Les durées dépendent de la charge du poste. Le bloc 6 réutilisait des log-Mel en cache ; "
               "la nouvelle extraction redécode les MP3 et recalcule le spectre nécessaire à HPSS. "
               "Les durées globales ne mesurent donc pas seulement le coût relatif des architectures."),
              (f"Les horodatages des caches comportent {len(timing['gaps_over_two_minutes_seconds'])} "
               f"intervalle(s) de plus de deux minutes sans nouvelle sauvegarde, avec un maximum de "
               f"{timing['largest_cache_write_gap_seconds'] / 60:.2f} minutes. La cause de ces grands "
               "écarts n’est pas enregistrée. La durée globale ne doit pas être présentée comme une "
               "mesure contrôlée de temps CPU continu ; le benchmark par fenêtre est plus informatif ici."),
              "", "## Limites et suite", "",
              ("Le ViT est préentraîné sur des images RGB : l’ordre original/H/P est une convention "
               "de transfert, sans apprentissage d’un agencement audio spécifique. HPSS distingue "
               "des structures durables et transitoires, sans isoler parfaitement instruments ou voix. "
               "Les résultats concernent cette configuration ; ils ne prouvent pas une amélioration "
               "générale de la compréhension de l’harmonie."), "",
              ("La validation a servi au choix de C et aux comparaisons ; aucun gain statistiquement "
               "significatif ni score final indépendant n’est revendiqué. Les deux extractions "
               "supplémentaires prévues sont consommées. Prochaine étape : consolider le protocole "
               "final, réaliser les trois graines CNN prévues et verrouiller les choix avant le test. "
               "Le test reste non évalué."), "",
              "## Reproduction et preuves", "", "Voir [le protocole et sa source HPSS](../docs/ABLATION_HPSS.md).", "",
              f"- [Rapport original](../{p0['folder']}/report.json)",
              f"- [Rapport HPSS](../{ph['folder']}/report.json)",
              "- [Comparaison vérifiable](VIT_HPSS_COMPARISON.json)",
              "- [Benchmark](VIT_HPSS_BENCHMARK.json)", ""]
    (ROOT / "reports/ABLATION_HPSS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
