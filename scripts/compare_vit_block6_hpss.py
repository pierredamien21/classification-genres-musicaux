"""Compare the combination with both single-factor references, including losses."""

import json
from pathlib import Path

from compare_vit_baseline import read_run, validate_comparable

from music_genre.data.audit import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    candidate = read_run("vit_block6_hpss")
    cp, cr, cf, rows, correct = candidate
    config = cf["provenance"]["config"]
    if config["feature_block"] != 6 or config["channels"] != "original_harmonic_percussive":
        raise ValueError("Wrong candidate representation")
    benchmark = ROOT / cf["provenance"]["benchmark"]["report"]
    if sha256_file(benchmark) != cf["benchmark_sha256"]:
        raise ValueError("Extraction benchmark changed")
    if any(t["warnings"] for t in cr["trials"]):
        raise ValueError("Candidate classifier emitted fit warnings")
    refs = {name: read_run(name) for name in ("vit_block6", "vit_hpss", "vit")}
    varying = {"vit_block6": {"name", "channels", "hpss"},
               "vit_hpss": {"name", "feature_block", "terminal_norm", "pooling"},
               "vit": {"name", "channels", "hpss", "feature_block", "terminal_norm", "pooling"}}
    comparisons = {}
    for name, (pointer, report, features, predictions, ref_correct) in refs.items():
        validate_comparable(report, cr, features, cf, predictions, rows)
        ignored = varying[name]
        if {k: v for k, v in config.items() if k not in ignored} != {
            k: v for k, v in features["provenance"]["config"].items() if k not in ignored}:
            raise ValueError("Uncontrolled representation difference")
        if name != "vit" and cf["provenance"]["parents"][name] != {
            "folder": report["feature_cache"], "report_sha256": report["feature_report_sha256"]}:
            raise ValueError("Comparator differs from extraction parent")
        comparisons[name] = {
            "reference_run": pointer,
            "macro_f1_difference": cr["validation"]["macro_f1"] - report["validation"]["macro_f1"],
            "accuracy_difference": cr["validation"]["accuracy"] - report["validation"]["accuracy"],
            "corrected": int((correct & ~ref_correct).sum()),
            "regressed": int((~correct & ref_correct).sum()),
            "both_correct": int((correct & ref_correct).sum()),
            "both_wrong": int((~correct & ~ref_correct).sum())}
    result = {"status": "complete", "candidate_run": cp, "comparisons": comparisons,
              "test_evaluated": False, "baseline_manifest_sha256": cr["baseline_manifest_sha256"],
              "index_sha256": cf["index_sha256"], "source_sha256": {p: sha256_file(ROOT / p) for p in (
                  "scripts/compare_vit_block6_hpss.py", "scripts/compare_vit_baseline.py")}}
    path = ROOT / "reports/VIT_BLOCK6_HPSS_COMPARISON.json"
    write_json(path, result)
    write_json(ROOT / "reports/latest_vit_block6_hpss_comparison.json", {
        "report": path.relative_to(ROOT).as_posix(), "report_sha256": sha256_file(path)})
    models = {**{n: r[1] for n, r in refs.items()}, "vit_block6_hpss": cr}
    lines = ["# Bloc 6 + HPSS : comparaison de validation", "",
             ("Même cohorte : 6 299 morceaux de train et 780 de validation, trois fenêtres identiques. "
             "Le test reste non évalué. Configurations et grille fixées avant les résultats."), "",
             "## Résultats", "", "| Modèle | C | Macro-F1 validation | Accuracy | Macro-F1 train |",
             "|---|---:|---:|---:|---:|"]
    for name, r in models.items():
        lines.append(f"| {name} | {r['selected_c']:g} | {r['validation']['macro_f1']:.4f} | "
                     f"{r['validation']['accuracy']:.4f} | {r['training']['macro_f1']:.4f} |")
    lines += ["", "## Différences candidat − référence", "",
              "| Référence | Différence macro-F1 | Corrections | Régressions |", "|---|---:|---:|---:|"]
    for name, c in comparisons.items():
        lines.append(f"| {name} | {c['macro_f1_difference']:+.4f} | {c['corrected']} | {c['regressed']} |")
    lines += ["", ("Contre bloc 6 original, seuls les canaux changent ; contre bloc 12 HPSS, seule la profondeur change. "
              "Le bloc 12 original est un repère à deux facteurs différents. Ces effets incluent la sélection de C "
              "dans la même grille pour chaque représentation, sans garantir la généralisation."), "",
              "## F1 par genre", "", "| Genre | " + " | ".join(models) + " |", "|---|" + "---:|"*len(models)]
    for label in cr["labels"]:
        lines.append(f"| {label} | " + " | ".join(f"{r['validation']['per_class'][label]['f1-score']:.4f}" for r in models.values()) + " |")
    lines += ["", "## Grille candidate", "", "| C | Macro-F1 | Accuracy | Itérations |", "|---:|---:|---:|---|"]
    for trial in cr["trials"]:
        lines.append(f"| {trial['c']:g} | {trial['validation']['macro_f1']:.4f} | {trial['validation']['accuracy']:.4f} | {trial['iterations']} |")
    lines += ["", "## Vérifications et limites", "",
              (f"Extraction : {cf['elapsed_seconds']/60:.2f} minutes dans l’invocation finale, "
              f"{cf['resumed_tracks']} morceaux repris ; ce temps exclut leur calcul antérieur. "
              f"Entraînement : {cr['elapsed_seconds']:.2f} s. Export : écart maximal {cr['export_max_probability_error']:.3g}."), "",
              ("Caches, modèles, prédictions, cohorte et index sont vérifiés ; les métriques sont recalculées. "
              "Aucun avertissement d’ajustement. La validation sert à sélectionner C et à comparer les variantes : "
              "ces scores ne sont pas une estimation indépendante ni une preuve de significativité."), "",
              ("[Rapport numérique](VIT_BLOCK6_HPSS_COMPARISON.json) · "
              "[Protocole](../docs/COMBINAISON_BLOC6_HPSS.md)"), ""]
    (ROOT / "reports/COMBINAISON_BLOC6_HPSS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"candidate": cr["validation"]["macro_f1"], "comparisons": comparisons}, indent=2), flush=True)


if __name__ == "__main__":
    main()
