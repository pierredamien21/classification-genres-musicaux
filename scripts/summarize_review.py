"""Publish a compact report of the technical selection and its remaining uncertainties."""

import csv
import json
from pathlib import Path

from music_genre.data.audit import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    pointer = json.loads((ROOT / "data/manifests/latest_baseline.json").read_text())
    folder = (ROOT / pointer["folder"]).resolve()
    if not folder.is_relative_to((ROOT / "data/manifests").resolve()):
        raise ValueError("Selection folder outside manifests")
    if sha256_file(folder / "report.json") != pointer["report_sha256"]:
        raise ValueError("Selection report changed")
    report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    for name, field in (("manifest.csv", "manifest_sha256"), ("decisions.csv", "decisions_sha256")):
        if sha256_file(folder / name) != report[field]:
            raise ValueError(f"Selection artifact changed: {name}")
    fp_folder = (ROOT / report["fingerprint_folder"]).resolve()
    if not fp_folder.is_relative_to((ROOT / "data/manifests").resolve()):
        raise ValueError("Fingerprint folder outside manifests")
    if sha256_file(fp_folder / "report.json") != report["fingerprint_report_sha256"]:
        raise ValueError("Fingerprint report changed")
    scan = json.loads((fp_folder / "report.json").read_text(encoding="utf-8"))
    controls = json.loads((ROOT / "reports/FINGERPRINT_CONTROLS.json").read_text(encoding="utf-8"))
    if (controls["policy_sha256"] != report["policy_sha256"]
            or controls["audit_sha256"] != report["audit_sha256"]):
        raise ValueError("Fingerprint controls belong to another policy or audit")
    detected_controls = sum(c["detected"] for c in controls["positive_controls"])
    total_controls = len(controls["positive_controls"])
    unexpected_controls = len(controls["unexpected_cross_track_matches"])
    with (folder / "decisions.csv").open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields, rows = reader.fieldnames, list(reader)
    with (folder / "pending_review.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(r for r in rows if r["decision"] == "hold")
    candidates = scan["candidates"]
    with (folder / "fingerprint_candidates.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[
            "a", "b", "shared_distinct_words", "offset_words", "overlap_words",
            "bit_error_rate", "exact_pcm", "cross_split", "same_genre",
        ])
        writer.writeheader()
        writer.writerows(candidates)
    counts = report["decision_counts"]
    prefix = "../" + folder.relative_to(ROOT).as_posix()
    lines = [
        "# Revue technique et sélection pour les premières expériences", "",
        f"Profil `{report['profile']}` appliqué le {report['created_at_utc'][:10]}.", "",
        (f"Sur 8 000 morceaux : **{counts.get('keep', 0)} retenus**, "
         f"**{counts.get('hold', 0)} en quarantaine réversible**, "
         f"**{counts.get('exclude', 0)} exclus de cette sélection**."),
        "", ("Il s’agit d’une revue automatisée suivant une politique documentée. "
        "Aucune écoute humaine ni résolution subjective des genres n’est revendiquée."),
        "", f"- [Manifeste pour les expériences]({prefix}/manifest.csv)",
        f"- [Journal des 8 000 décisions]({prefix}/decisions.csv)",
        f"- [Cas encore en attente]({prefix}/pending_review.csv)",
        f"- [Paires candidates détaillées]({prefix}/fingerprint_candidates.csv)",
        f"- [Provenance complète]({prefix}/report.json)", "",
        "## Politique appliquée", "",
        "- Erreurs du décodeur de référence : exclusion technique pour ce pipeline.",
        "- Diagnostics, silence majoritaire, licence manquante ou empreinte insuffisante : quarantaine.",
        ("- Alerte de niveau seule : conservation avec son indicateur ; aucune normalisation "
        "ou correction destructive du signal."),
        ("- Sons exactement identiques, même genre et même partition : plus petit identifiant "
        "éligible conservé, répétitions exclues avec lien vers le représentant."),
        "- Conflit de genres sur un contenu identique : groupe isolé, sans réétiquetage arbitraire.",
        ("- Candidats non exacts : composante entière mise en quarantaine, sans les déclarer "
        "doublons confirmés."),
        "", ("Les fichiers et partitions d’origine sont inchangés. La sélection des données ne "
        "repose sur aucun score de modèle. Les futures performances devront être annoncées "
        "comme celles de ce sous-ensemble filtré, avec ses effectifs, et non comme un résultat "
        "directement comparable à FMA-small intégral."),
        "", "## Effectifs retenus", "", "| Genre | Train | Validation | Test |",
        "|---|---:|---:|---:|",
    ]
    for genre, values in report["selected_by_genre_and_split"].items():
        lines.append(f"| {genre} | {values.get('train', 0)} | "
                     f"{values.get('validation', 0)} | {values.get('test', 0)} |")
    values = report["selected_by_split"]
    lines.append(f"| **Total** | **{values.get('train', 0)}** | "
                 f"**{values.get('validation', 0)}** | **{values.get('test', 0)}** |")
    nonexact = [c for c in candidates if not c["exact_pcm"]]
    lines += [
        "", "## Empreintes perceptuelles", "",
        f"Outil vérifié : `{scan['tool_version']}`. Les 8 000 fichiers ont été soumis à l’outil.",
        (f"{len(scan['nonzero_exit_ids'])} sorties en erreur ; "
        f"{len(scan['low_information_ids'])} empreintes trop courtes ou peu informatives "
        "(catégories pouvant se recouvrir)."),
        (f"{len(candidates)} paires candidates, dont {len(nonexact)} non identiques au niveau PCM "
        f"et {sum(c['cross_split'] for c in nonexact)} non exactes entre partitions."),
        "", ("La recherche utilise des mots Chromaprint communs peu fréquents, puis compare les "
        "bits après alignement temporel. Seuils fixés avant la recherche : dix mots distincts "
        "communs, au moins 120 mots alignés, au plus 10 % de bits différents. Les mots présents "
        "dans plus de 25 morceaux sont ignorés à la recherche. Ces seuils sont des heuristiques "
        "locales ; ils ne reproduisent pas le service AcoustID."),
        "", (f"Contrôle positif : {detected_controls}/{total_controls} transformations de "
        f"{len(controls['training_track_ids'])} morceaux de **training** retrouvées "
        "(mono 16 kHz avec gain réduit ou décalage de deux secondes) ; "
        f"{unexpected_controls} rapprochement(s) entre sources distinctes. "
        "[Résultats des contrôles](FINGERPRINT_CONTROLS.json). "
        "Ce petit contrôle ne mesure pas le rappel sur tout le dataset."),
        "", "### Candidats non exacts (au plus trente affichés)", "",
        "| A | B | Entre partitions | Taux de bits différents |", "|---|---|---|---:|",
    ]
    for item in sorted(nonexact, key=lambda c: (not c["cross_split"], c["bit_error_rate"]))[:30]:
        lines.append(f"| {item['a']} | {item['b']} | "
                     f"{'oui' if item['cross_split'] else 'non'} | {item['bit_error_rate']:.4f} |")
    lines += [
        "", "## Vérifications et limites", "",
        ("Le constructeur vérifie les empreintes de l’audit, des signatures et de la politique, "
        "l’identité des listes de morceaux, l’absence de doublons PCM retenus et l’absence "
        "d’artiste ou de groupe de contenu retenu dans plusieurs partitions."),
        "", ("La paire exacte **057371 / 072149** porte deux genres différents "
        "(Instrumental / Experimental) : elle reste isolée. Les six échecs initiaux du décodeur "
        "sont également analysés par fpcalc/FFmpeg ; voir les diagnostics du scan."),
        "", ("Les empreintes peuvent manquer des réencodages fortement altérés, des modifications "
        "de tempo, des recouvrements courts ou des extraits non recouvrants du même morceau. "
        "Une sortie normale du second outil ne prouve pas qu’un MP3 soit intact. "
        "Les avertissements du premier décodeur restent donc pris en compte."),
        "", ("La bibliothèque [Chromaprint](https://github.com/acoustid/chromaprint) vise les sons "
        "presque identiques ; son [outil fpcalc](https://acoustid.org/chromaprint) calcule les "
        "empreintes localement. Aucune donnée audio ou empreinte n’a été téléversée."),
        "", "## Prochaine étape", "",
        ("Extraire les descripteurs de timbre, rythme et harmonie sur ce manifeste versionné, "
        "puis entraîner une première référence. Ajuster toute normalisation sur train seulement, "
        "sélectionner sur validation et garder test pour l’évaluation finale. "
        "La file de quarantaine pourra faire l’objet d’une revue ultérieure et d’une analyse "
        "de sensibilité documentée."), "",
    ]
    (ROOT / "reports/REVUE_DONNEES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Report: {ROOT / 'reports/REVUE_DONNEES.md'}")


if __name__ == "__main__":
    main()
