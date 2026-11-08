"""Audit paired validation regressions without re-fitting or reading test audio."""

import csv
import json
from collections import Counter
from pathlib import Path

from compare_vit_baseline import read_run, validate_comparable

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.errors import align_validation, genre_changes, paired_outcomes

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE = "vit_block6_hpss"
REFERENCES = ("vit_hpss", "vit_block6")


def main():
    runs = {name: read_run(name) for name in (CANDIDATE, *REFERENCES)}
    cp, cr, cf, rows, _ = runs[CANDIDATE]
    baseline = json.loads((ROOT / "data/manifests/latest_baseline.json").read_text("utf-8"))
    manifest = (ROOT / baseline["folder"] / "manifest.csv").resolve()
    if (not manifest.is_relative_to((ROOT / "data/manifests").resolve())
            or sha256_file(manifest) != baseline["manifest_sha256"]
            or baseline["manifest_sha256"] != cr["baseline_manifest_sha256"]):
        raise ValueError("Manifest identity changed")
    with manifest.open(encoding="utf-8", newline="") as stream:
        selected = [row for row in csv.DictReader(stream) if row["split"] == "validation"]
    sources = {row["track_id"]: row for row in selected}
    labels = cr["labels"]
    ids, truth, matrices = align_validation(selected, {n: r[3] for n, r in runs.items()}, labels)
    predictions = {name: matrix.argmax(axis=1) for name, matrix in matrices.items()}
    candidate = predictions[CANDIDATE]
    if any(not sources[key]["artist_id"] for key in ids):
        raise ValueError("Missing artist identity")
    comparisons, cases = {}, []
    lines = ["# Corrections et régressions de bloc 6 + HPSS", "",
             ("Analyse descriptive des 780 morceaux de validation, sans nouvel entraînement. "
             "Les étiquettes sont celles du manifeste ; aucune cause acoustique n’est déduite "
             "des seules prédictions. Le test reste non évalué."), ""]
    for name in REFERENCES:
        pointer, report, features, ref_rows, _ = runs[name]
        validate_comparable(report, cr, features, cf, ref_rows, rows)
        reference = predictions[name]
        paired = paired_outcomes(truth, reference, candidate)
        by_genre = genre_changes(truth, reference, candidate, labels)
        artists = {"corrected": Counter(), "regressed": Counter()}
        for i, key in enumerate(ids):
            ref_ok, new_ok = reference[i] == truth[i], candidate[i] == truth[i]
            if ref_ok == new_ok:
                continue
            outcome = "corrected" if new_ok else "regressed"
            artist = sources[key]["artist_id"]
            artists[outcome][artist] += 1
            cases.append({"reference": name, "outcome": outcome, "track_id": key,
                          "artist_id": artist, "true_label": labels[int(truth[i])],
                          "reference_prediction": labels[int(reference[i])],
                          "candidate_prediction": labels[int(candidate[i])]})
        for outcome, counts in artists.items():
            if sum(counts.values()) != paired[outcome]:
                raise ValueError("Artist totals disagree with paired results")
        comparisons[name] = {"reference_run": pointer, "paired": paired,
                             "by_genre": by_genre, "artist_counts": artists}
        lines += [f"## Face à {name}", "",
                  (f"{paired['corrected']} corrections, {paired['regressed']} régressions ; "
                  f"{paired['both_wrong']} morceaux restent mal classés par les deux modèles."), "",
                  "| Genre réel | Effectif | Corrections | Régressions | F1 référence | F1 candidat | FP référence → candidat |",
                  "|---|---:|---:|---:|---:|---:|---:|"]
        for label, item in by_genre.items():
            lines.append(f"| {label} | {item['tracks']} | {item['corrected']} | {item['regressed']} | "
                         f"{report['validation']['per_class'][label]['f1-score']:.4f} | "
                         f"{cr['validation']['per_class'][label]['f1-score']:.4f} | "
                         f"{item['reference_false_positives']} → {item['candidate_false_positives']} |")
        lines += ["", ("FP : morceaux d’autres genres attribués à ce genre. "
                  "Une F1 peut augmenter sans gain de rappel si les faux positifs diminuent."), "",
                  (f"Corrections réparties sur {len(artists['corrected'])} artistes ; "
                  f"régressions sur {len(artists['regressed'])} artistes. "
                  f"Maximum de régressions pour un artiste : {max(artists['regressed'].values(), default=0)}. "
                  "Ces comptes ne remplacent pas une estimation d’incertitude groupée par artiste."), ""]
    shared_errors = (candidate != truth)
    for name in REFERENCES:
        shared_errors &= predictions[name] != truth
    shared = {label: int((shared_errors & (truth == i)).sum()) for i, label in enumerate(labels)}
    case_path = ROOT / "reports/BLOCK6_HPSS_CHANGED_TRACKS.csv"
    with case_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["reference", "outcome", "track_id", "artist_id",
                                                   "true_label", "reference_prediction", "candidate_prediction"])
        writer.writeheader()
        writer.writerows(cases)
    output = {"candidate_run": cp, "baseline": baseline, "test_evaluated": False,
              "validation_tracks": len(ids), "validation_artists": len({r['artist_id'] for r in selected}),
              "comparisons": comparisons, "all_three_wrong_by_genre": shared,
              "changed_tracks": {"path": case_path.relative_to(ROOT).as_posix(),
                                 "sha256": sha256_file(case_path)},
              "sources": {p: sha256_file(ROOT / p) for p in (
                  "scripts/analyze_block6_hpss_errors.py", "scripts/compare_vit_baseline.py",
                  "src/music_genre/evaluation/errors.py", "src/music_genre/evaluation/metrics.py")}}
    path = ROOT / "reports/BLOCK6_HPSS_ERRORS.json"
    write_json(path, output)
    write_json(ROOT / "reports/latest_block6_hpss_errors.json", {
        "report": path.relative_to(ROOT).as_posix(), "report_sha256": sha256_file(path)})
    lines += ["## Erreurs communes aux trois modèles", "",
              "Bloc 6 HPSS, bloc 12 HPSS et bloc 6 original : "
              + "; ".join(f"{label} {count}" for label, count in shared.items()) + ".", "",
              "## Traçabilité et limites", "",
              ("Les modèles, caches, index et prédictions sont vérifiés par empreintes ; "
              "les métriques sont recalculées et les identifiants réconciliés avec le manifeste. "
              "La validation a guidé la sélection des variantes : aucun test de significativité "
              "ni intervalle de généralisation n’est présenté ici. Écoute subjective non réalisée."), "",
              "Reproduction : `.\\.venv\\Scripts\\python.exe scripts/analyze_block6_hpss_errors.py`.", "",
              "- [Rapport numérique et empreintes](BLOCK6_HPSS_ERRORS.json)",
              "- [Tous les morceaux corrigés ou dégradés](BLOCK6_HPSS_CHANGED_TRACKS.csv)",
              "- [Décision et prochaines étapes](../docs/DECISIONS_EXPERIMENTALES.md)", ""]
    (ROOT / "reports/REGRESSIONS_BLOC6_HPSS.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"comparisons": comparisons, "shared_errors": shared}, indent=2))


if __name__ == "__main__":
    main()
