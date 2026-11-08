"""Create a readable audit report and a review queue from verified local outputs."""

import csv
import json
from pathlib import Path

import numpy as np

from music_genre.data.audit import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    pointer = json.loads((ROOT / "data/manifests/latest_audit.json").read_text(encoding="utf-8"))
    folder = (ROOT / pointer["folder"]).resolve()
    if not folder.is_relative_to((ROOT / "data/manifests").resolve()):
        raise ValueError("Audit folder is outside manifests")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    manifest = folder / "manifest.csv"
    if sha256_file(manifest) != pointer["manifest_sha256"] or (
        report["manifest_sha256"] != pointer["manifest_sha256"]
    ):
        raise ValueError("Manifest checksum mismatch")
    with manifest.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)
    if len(rows) != report["tracks"] or len({r["track_id"] for r in rows}) != len(rows):
        raise ValueError("Manifest record count mismatch")
    pending = [r for r in rows if r["status"] != "valid"]
    with (folder / "review_queue.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(pending)
    path_prefix = "../" + folder.relative_to(ROOT).as_posix()
    counts = report["status_counts"]
    lines = [
        "# Audit technique de FMA-small", "",
        f"Exécuté le {report['created_at_utc'][:10]} ; durée {report['elapsed_seconds'] / 60:.1f} minutes.",
        "", (f"**{len(rows):,} morceaux contrôlés** : {counts.get('valid', 0)} sans alerte selon "
        f"ces contrôles, {counts.get('needs_review', 0)} à examiner, "
        f"{counts.get('excluded', 0)} exclus pour défaut bloquant."), "",
        ("Les fichiers originaux et les partitions officielles sont conservés. « Sans alerte » "
        "ne garantit ni l’absence de défaut audible ni l’absence de doublon perceptuel."),
        "", f"- [Manifeste complet]({path_prefix}/manifest.csv)",
        f"- [File de revue, exclusions comprises]({path_prefix}/review_queue.csv)",
        f"- [Provenance et résultats détaillés]({path_prefix}/report.json)",
        "", f"SHA-256 du manifeste : `{report['manifest_sha256']}`.", "",
        "## Contrôles et interprétation", "",
        ("Lecture intégrale par libsndfile, blocs d’une seconde, fréquence et canaux d’origine, "
        "sans rééchantillonnage. Empreinte du fichier et empreinte du PCM float32 little-endian "
        "incluant fréquence et nombre de canaux. Diagnostics natifs capturés par fichier."),
        "", ("Seuils fixés avant l’audit dans `configs/audit.json` : durée hors de 29–31 s signalée ; "
        "blocs RMS ≤ −60 dBFS considérés silencieux et revue à partir de 50 % de durée silencieuse ; "
        "revue si au moins 1 % des échantillons ont une amplitude absolue ≥ 0,999. "
        "Cette dernière mesure indique une proximité de la pleine échelle, pas une preuve de saturation."),
        "", ("Exclusion technique uniquement si décodage impossible, signal vide/non fini "
        "ou entièrement nul. Les autres alertes, y compris les doublons, restent en revue. "
        "Aucun genre, score de modèle ou résultat de test n’intervient dans ces critères."),
        "", "## Répartition des statuts", "",
        "| Partition | Sans alerte | À examiner | Exclus |", "|---|---:|---:|---:|",
    ]
    for split, values in report["split_status_counts"].items():
        lines.append(f"| {split} | {values.get('valid', 0)} | "
                     f"{values.get('needs_review', 0)} | {values.get('excluded', 0)} |")
    lines += ["", "| Genre | Sans alerte | À examiner | Exclus |", "|---|---:|---:|---:|"]
    for genre, values in report["genre_status_counts"].items():
        lines.append(f"| {genre} | {values.get('valid', 0)} | "
                     f"{values.get('needs_review', 0)} | {values.get('excluded', 0)} |")
    lines += ["", "## Alertes", "", "Un morceau peut cumuler plusieurs alertes.", "",
              "| Alerte | Morceaux |", "|---|---:|"]
    for flag, count in sorted(report["flag_counts"].items()):
        lines.append(f"| {flag} | {count} |")
    durations = [float(r["duration_seconds"]) for r in rows if r["decode_status"] == "ok"]
    if durations:
        low, median, high = np.quantile(durations, [0, 0.5, 1])
        lines += ["", (f"Durées décodées : minimum {low:.3f} s, médiane {median:.3f} s, "
                  f"maximum {high:.3f} s. La durée des métadonnées peut décrire le morceau "
                  "d’origine et ne remplace pas celle de l’extrait.")]
    lines += ["", "## Échecs de décodage", "",
              "| Morceau | Motif |", "|---|---|"]
    for item in report["excluded"]:
        lines.append(f"| {item['track_id']} | {item['reason']} |")
    recheck_path = folder / "failure_recheck.json"
    if recheck_path.is_file():
        recheck = json.loads(recheck_path.read_text(encoding="utf-8"))
        if recheck["manifest_sha256"] != pointer["manifest_sha256"]:
            raise ValueError("Failure recheck belongs to another manifest")
        if all(c["exists"] and c["matches_archive_and_first_audit"]
               and c["repeat_decode_status"] == "failed" for c in recheck["checks"]):
            lines += ["", (f"Les {len(recheck['checks'])} échecs ont été reproduits en lecture "
                      "séquentielle. Les fichiers existent et leurs octets correspondent à "
                      "l’archive d’origine. Le message « File does not exist » de libsndfile "
                      "ne décrit donc pas ici une absence réelle de fichier. Cela ne prouve "
                      "pas qu’aucun autre décodeur puisse récupérer un signal."),
                      f"[Vérification détaillée]({path_prefix}/failure_recheck.json)."]
    benchmark_path = ROOT / "reports/environment_benchmark.json"
    if benchmark_path.is_file():
        benchmark = json.loads(benchmark_path.read_text(encoding="utf-8"))
        sample_ids = {r["track_id"] for r in benchmark["samples"]}
        flagged = [r["track_id"] for r in rows
                   if r["track_id"] in sample_ids and r["decoder_diagnostic"]]
        lines += ["", "Morceaux du premier benchmark avec diagnostic natif : "
                  + (", ".join(flagged) or "aucun") + "."]
    lines += ["", "## Doublons et fuite entre partitions", ""]
    for field, groups in report["duplicates"].items():
        lines.append(f"- `{field}` : {len(groups)} groupe(s), dont "
                     f"{sum(g['cross_split'] for g in groups)} entre partitions.")
        for group in groups:
            lines.append(f"  - Morceaux {', '.join(group['track_ids'])} ; "
                         f"partitions {', '.join(group['splits'])}.")
    lines += ["", (f"Identifiants d’artiste partagés entre partitions : "
              f"{sum(len(v) for v in report['artist_overlap'].values())} "
              "(somme des intersections deux à deux)."),
              "", ("Les empreintes exactes ne détectent pas toutes les reprises, versions "
              "réencodées, changements de volume ou extraits décalés. Une recherche perceptuelle "
              "reste à mener avant de déclarer l’absence de fuite de contenu."), "",
              "## Licences et publication", "",
              ("La licence de chaque morceau est reprise des métadonnées dans le manifeste. "
              "Ce relevé ne vaut pas validation des droits de redistribution ; "
              "ne pas publier les fichiers audio avec le portfolio."), "",
              "## Suite", "",
              ("Examiner la file de revue, comparer au besoin les fichiers problématiques avec "
              "un second décodeur, puis consigner les décisions dans un journal distinct. "
              "Construire ensuite un manifeste d’entraînement approuvé, sans modifier les originaux "
              "ni déplacer un morceau vers une autre partition. Les alertes de durée ou de niveau "
              "ne justifient pas à elles seules une suppression."), ""]
    output = ROOT / "reports/AUDIT_DONNEES.md"
    output.write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {output}")
    print(f"Review queue: {folder / 'review_queue.csv'}")


if __name__ == "__main__":
    main()
