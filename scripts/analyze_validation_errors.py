"""Compare saved validation errors and create an unannotated local listening packet."""

import base64
import csv
import html
import json
import os
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path

import numpy as np

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.errors import align_validation, paired_outcomes, select_review
from music_genre.evaluation.metrics import classification_metrics

ROOT = Path(__file__).resolve().parents[1]
SYSTEMS = {"vit": "ViT bloc 12 original", "vit_block6": "ViT bloc 6 original",
           "vit_hpss": "ViT bloc 12 HPSS", "descriptor": "Descripteurs",
           **{f"cnn20_seed{s}": f"CNN prolongé, graine {s}" for s in (42, 43, 44)}}
STRATA = {
    "pop_shared_vit_error": "Pop manquée par les trois ViT",
    "pop_rescued_block6": "Pop manquée par HPSS, retrouvée au bloc 6",
    "pop_hpss_correct": "Pop correctement reconnue par HPSS",
    "pop_false_positive": "Faux positif Pop sous HPSS",
    "hiphop_hpss_regression": "Hip-Hop correcte avant HPSS, perdue après",
    "hiphop_hpss_correction": "Hip-Hop retrouvée avec HPSS",
    "hiphop_both_wrong": "Hip-Hop manquée avant et après HPSS",
}


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def load_inputs():
    pointer = json.loads((ROOT / "data/manifests/latest_baseline.json").read_text(encoding="utf-8"))
    manifest = (ROOT / pointer["folder"] / "manifest.csv").resolve()
    if (not manifest.is_relative_to((ROOT / "data/manifests").resolve())
            or sha256_file(manifest) != pointer["manifest_sha256"]):
        raise ValueError("Manifest changed")
    selected = [row for row in read_csv(manifest) if row["split"] == "validation"]
    labels = json.loads((ROOT / "configs/labels.json").read_text(encoding="utf-8"))["labels"]
    predictions, reports, pointers = {}, {}, {}
    for name in SYSTEMS:
        run_pointer = json.loads((ROOT / f"reports/latest_{name}_run.json").read_text(encoding="utf-8"))
        folder = (ROOT / run_pointer["folder"]).resolve()
        if (not folder.is_relative_to((ROOT / "runs").resolve())
                or sha256_file(folder / "report.json") != run_pointer["report_sha256"]):
            raise ValueError("Run report changed")
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        if (report["test_loaded"] or report["test_evaluated"] or report["labels"] != labels
                or report["baseline_manifest_sha256"] != pointer["manifest_sha256"]):
            raise ValueError("Incomparable run")
        model = "model.safetensors" if name.startswith("cnn") else "model.npz"
        for filename, field in ((model, "model_sha256"), ("validation_predictions.csv", "validation_predictions_sha256")):
            if sha256_file(folder / filename) != report[field]:
                raise ValueError("Run artifact changed")
        predictions[name] = read_csv(folder / "validation_predictions.csv")
        reports[name], pointers[name] = report, run_pointer
    ids, truth, matrices = align_validation(selected, predictions, labels)
    for name, matrix in matrices.items():
        if classification_metrics(truth, matrix, labels) != reports[name]["validation"]:
            raise ValueError("Metrics differ from saved report")
    return pointer, {r["track_id"]: r for r in selected}, labels, ids, truth, matrices, reports, pointers


def metadata_for(ids):
    path = ROOT / "data/raw/fma_metadata/tracks.csv"
    wanted, found = set(ids), {}
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        columns = list(zip(next(reader), next(reader), strict=True))
        next(reader)
        for row in reader:
            key = row[0].zfill(6)
            if key in wanted:
                item = dict(zip(columns, row, strict=True))
                found[key] = {"title": item[("track", "title")], "artist": item[("artist", "name")],
                              "artist_id": item[("artist", "id")], "genre": item[("track", "genre_top")],
                              "split": item[("set", "split")], "genres": item[("track", "genres")],
                              "genres_all": item[("track", "genres_all")], "language": item[("track", "language_code")]}
    if set(found) != wanted:
        raise ValueError("Missing review metadata")
    return found, sha256_file(path)


def make_packet(folder, selected, sources, labels, ids, truth, matrices):
    metadata, metadata_sha = metadata_for([r["track_id"] for r in selected])
    positions = {key: i for i, key in enumerate(ids)}
    queue, cards = [], []
    for number, item in enumerate(selected, 1):
        key = item["track_id"]
        source, meta, pos = sources[key], metadata[key], positions[key]
        if meta["artist_id"] != source["artist_id"] or meta["genre"] != source["genre"] or meta["split"] != "validation":
            raise ValueError("Original metadata differs from manifest")
        audio_path = (ROOT / "data/raw" / source["audio_path"]).resolve()
        if (not audio_path.is_relative_to((ROOT / "data/raw/fma_small").resolve())
                or sha256_file(audio_path) != source["source_sha256"]):
            raise ValueError("Review audio changed or escaped dataset")
        predictions = {name: {"label": labels[int(p[pos].argmax())], "max_probability": float(p[pos].max()),
                              "true_label_probability": float(p[pos, truth[pos]])} for name, p in matrices.items()}
        queue.append({**item, "artist_id": source["artist_id"], "metadata": meta, "truth": source["genre"],
                      "split": "validation", "audio_path": audio_path.relative_to(ROOT).as_posix(),
                      "source_sha256": source["source_sha256"], "license": source["license"],
                      "predictions": predictions, "review_status": "pending", "observations": ""})
        rows = "".join(f"<tr><td>{html.escape(SYSTEMS[name])}</td><td>{html.escape(p['label'])}</td>"
                       f"<td>{p['max_probability']:.3f}</td></tr>" for name, p in predictions.items())
        encoded = base64.b64encode(audio_path.read_bytes()).decode("ascii")
        cards.append(f'''<article class="case" data-track="{key}"><h2>Extrait {number:02d}</h2>
<audio controls preload="none" src="data:audio/mpeg;base64,{encoded}"></audio>
<label>Observations d’écoute, genres plausibles et incertitudes<textarea aria-label="Observations extrait {number:02d}"></textarea></label>
<label><input type="checkbox"> Écoute et revue réalisées</label>
<details><summary>Afficher l’étiquette, les modèles et les métadonnées</summary>
<p><strong>{html.escape(meta['title'])}</strong> — {html.escape(meta['artist'])} · piste {key} · artiste {source['artist_id']}<br>
Étiquette FMA : <strong>{html.escape(source['genre'])}</strong><br>
Sélection : {html.escape(STRATA[item['stratum']])}</p>
<table><thead><tr><th>Modèle</th><th>Prédiction</th><th>Probabilité maximale</th></tr></thead><tbody>{rows}</tbody></table>
<p class="subtle">Ces probabilités ne sont pas calibrées. Une valeur élevée ne confirme pas une erreur de label.<br>
Genres FMA détaillés (identifiants) : {html.escape(meta['genres'])}. Langue déclarée : {html.escape(meta['language']) or 'non renseignée'}.<br>
Licence : {html.escape(source['license'])}. SHA-256 : <code>{source['source_sha256']}</code>.</p></details></article>''')
    template = (ROOT / "templates/error_review.html").read_text(encoding="utf-8")
    page = template.replace("__PACKET__", folder.name).replace("__COUNT__", str(len(queue))).replace("__CARDS__", "\n".join(cards))
    (folder / "index.html").write_text(page, encoding="utf-8")
    write_json(folder / "queue.json", queue)
    write_json(folder / "annotations_template.json", {"schema_version": 1, "packet": folder.name,
               "annotations": [{"track_id": q["track_id"], "status": "pending", "observations": ""} for q in queue]})
    return queue, metadata_sha


def plot_confusions(reports, labels):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/interim/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.3), layout="constrained")
    for ax, name in zip(axes, ("vit", "vit_block6", "vit_hpss"), strict=True):
        counts = np.array(reports[name]["validation"]["confusion_matrix"])
        values = counts / counts.sum(axis=1, keepdims=True)
        im = ax.imshow(values, cmap="Blues", vmin=0, vmax=1)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(counts[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if values[i, j] > 0.5 else "black")
        ax.set_xticks(range(len(labels)), labels, rotation=65, ha="right", fontsize=9)
        ax.set_yticks(range(len(labels)), labels, fontsize=9)
        ax.set(title=SYSTEMS[name], xlabel="Genre prédit", ylabel="Genre FMA")
    fig.colorbar(im, ax=axes, shrink=0.75, label="Proportion dans le genre réel")
    fig.suptitle("Validation : mêmes 780 morceaux — nombres dans les cellules, couleurs par ligne")
    target = ROOT / "reports/figures/validation_confusions"
    target.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(target.with_suffix('.' + ext), dpi=150)
    plt.close(fig)
    return {ext: sha256_file(target.with_suffix('.' + ext)) for ext in ("png", "pdf")}


def main():
    baseline, sources, labels, ids, truth, matrices, reports, pointers = load_inputs()
    config_path = ROOT / "configs/error_review.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config["split"] != "validation" or set(config["strata"]) != set(STRATA):
        raise ValueError("Unsupported review protocol")
    predicted = {name: p.argmax(axis=1) for name, p in matrices.items()}
    pairs = {}
    for a, b in combinations(SYSTEMS, 2):
        pairs[f"{a}__{b}"] = {"reference": a, "candidate": b,
            "all": paired_outcomes(truth, predicted[a], predicted[b]),
            "by_genre": {label: paired_outcomes(truth[truth == i], predicted[a][truth == i], predicted[b][truth == i])
                         for i, label in enumerate(labels)}}
    pop, hip = labels.index("Pop"), labels.index("Hip-Hop")
    a, b, h = predicted["vit"], predicted["vit_block6"], predicted["vit_hpss"]
    masks = {"pop_shared_vit_error": (truth == pop) & (a != truth) & (b != truth) & (h != truth),
             "pop_rescued_block6": (truth == pop) & (h != truth) & (b == truth),
             "pop_hpss_correct": (truth == pop) & (h == truth),
             "pop_false_positive": (truth != pop) & (h == pop),
             "hiphop_hpss_regression": (truth == hip) & (a == truth) & (h != truth),
             "hiphop_hpss_correction": (truth == hip) & (a != truth) & (h == truth),
             "hiphop_both_wrong": (truth == hip) & (a != truth) & (h != truth)}
    strata = {name: [key for key, keep in zip(ids, mask, strict=True) if keep] for name, mask in masks.items()}
    chosen, coverage = select_review(strata, sources, config["per_stratum"], config["seed"])
    folder = ROOT / "data/interim" / ("error-review-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    folder.mkdir(parents=True, exist_ok=False)
    queue, metadata_sha = make_packet(folder, chosen, sources, labels, ids, truth, matrices)
    counts = {label: {"tracks": int((truth == i).sum()),
                     "artists": len({sources[key]["artist_id"] for key, t in zip(ids, truth, strict=True) if t == i})}
              for i, label in enumerate(labels)}
    result = {"schema_version": 1, "baseline": baseline, "runs": pointers, "labels": labels,
              "validation_counts": counts, "pairs": pairs, "review_coverage": coverage,
              "review_folder": folder.relative_to(ROOT).as_posix(), "review_count": len(queue),
              "reviewed_count": 0, "human_listening_performed": False, "test_audio_loaded": False,
              "test_evaluated": False, "config_sha256": sha256_file(config_path), "metadata_sha256": metadata_sha,
              "packet_sha256": {name: sha256_file(folder / name) for name in ("queue.json", "index.html", "annotations_template.json")},
              "figure_sha256": plot_confusions(reports, labels),
              "source_sha256": {name: sha256_file(ROOT / name) for name in (
                  "scripts/analyze_validation_errors.py", "src/music_genre/evaluation/errors.py", "templates/error_review.html")}}
    write_json(ROOT / "reports/VALIDATION_ERRORS.json", result)
    write_json(ROOT / "reports/latest_validation_errors.json", {
        "report": "reports/VALIDATION_ERRORS.json", "report_sha256": sha256_file(ROOT / "reports/VALIDATION_ERRORS.json")})
    lines = ["# Erreurs de validation : Pop, Hip-Hop et désaccords", "",
             ("Analyse automatisée terminée ; revue d’écoute préparée mais **non réalisée**. "
             "Les métriques et prédictions sont vérifiées sur les mêmes 780 morceaux. Test non évalué."), "",
             "## Précision et rappel", "",
             "| Modèle | Genre | Support | Artistes | Précision | Rappel | F1 |",
             "|---|---|---:|---:|---:|---:|---:|"]
    for name, title in SYSTEMS.items():
        for label in ("Pop", "Hip-Hop"):
            m = reports[name]["validation"]["per_class"][label]
            lines.append(f"| {title} | {label} | {int(m['support'])} | {counts[label]['artists']} | "
                         f"{m['precision']:.4f} | {m['recall']:.4f} | {m['f1-score']:.4f} |")
    lines += ["", "CNN : trois réseaux distincts sont affichés ; aucun ensemble ni meilleure graine sélectionnée.", "",
              "## Ce que HPSS corrige ou dégrade face au bloc 12 original", "",
              "| Ensemble | Les deux corrects | Corrigés par HPSS | Perdus avec HPSS | Les deux faux | Désaccords |",
              "|---|---:|---:|---:|---:|---:|"]
    pair = pairs["vit__vit_hpss"]
    for name, value in [("Tous", pair["all"]), *pair["by_genre"].items()]:
        lines.append(f"| {name} | {value['both_correct']} | {value['corrected']} | {value['regressed']} | "
                     f"{value['both_wrong']} | {value['disagreements']} |")
    lines += ["", "Deux modèles peuvent se tromper avec des classes différentes : un désaccord n’est pas nécessairement une correction.", "",
              "## Vers quels genres vont les morceaux Pop et Hip-Hop sous HPSS ?", "",
              "| Genre réel | Electronic | Experimental | Folk | Hip-Hop | Instrumental | International | Pop | Rock |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for label in ("Pop", "Hip-Hop"):
        row = reports["vit_hpss"]["validation"]["confusion_matrix"][labels.index(label)]
        lines.append(f"| {label} | " + " | ".join(map(str, row)) + " |")
    lines += ["", "![Confusions ViT](figures/validation_confusions.png)", "",
              "[Figure PDF](figures/validation_confusions.pdf). Couleurs normalisées par genre réel, nombres bruts dans les cellules.", "",
              "## Échantillon de revue", "",
              f"**{len(queue)} extraits, {len({q['artist_id'] for q in queue})} artistes distincts, aucune annotation d’écoute réalisée.**", "",
              "| Catégorie | Candidats | Sélectionnés | Manquants au quota |", "|---|---:|---:|---:|"]
    for name, value in coverage.items():
        lines.append(f"| {STRATA[name]} | {value['available_tracks']} | {value['selected']} | {value['shortfall']} |")
    prefix = "../" + folder.relative_to(ROOT).as_posix()
    lines += ["", f"- [Ouvrir les fiches d’écoute]({prefix}/index.html)",
              f"- [Sélection et provenance]({prefix}/queue.json)",
              f"- [Modèle d’annotations vierge]({prefix}/annotations_template.json)", "",
              ("Sélection déterministe par empreinte avec graine fixe, catégories rares d’abord et au plus un morceau par artiste. "
               "Les cas sont ciblés pour comprendre des phénomènes ; ils ne constituent pas un échantillon représentatif des taux d’erreur. "
               "La catégorie « manquée par les trois ViT » n’exige pas qu’ils prédisent tous le même mauvais genre."), "",
              "## Interprétation et suite", "",
              ("La faible F1 Pop combine des faux négatifs et des faux positifs. Les confusions et désaccords localisent "
               "les erreurs, mais ne démontrent ni une mauvaise annotation ni une cause acoustique. Les probabilités maximales "
               "ne sont pas calibrées. Les genres FMA détaillés et la langue déclarée sont des indices de revue, pas des vérités auditives."), "",
              ("Écouter avant d’afficher les prédictions ; décrire voix, rythme, timbres, évolution de l’extrait et ambiguïtés. "
               "Exporter les observations JSON. Aucune modification automatique des labels, partitions ou originaux. "
               "Une éventuelle nouvelle expérience sera motivée par des observations documentées ; bloc 6 + HPSS reste non testé."), "",
              "## Reproduction", "", "Commande : `.\\.venv\\Scripts\\python.exe scripts/analyze_validation_errors.py`.", "",
              ("Chaque exécution crée un nouveau dossier d’écoute et conserve les précédents. Les annotations exportées doivent "
               "être conservées séparément ; la synthèse automatisée ne les marque jamais comme réalisées."), "",
              "[Résultats détaillés, toutes les paires et empreintes](VALIDATION_ERRORS.json).", ""]
    (ROOT / "reports/ERREURS_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"pair_hpss": pair, "review_coverage": coverage, "folder": str(folder)}, indent=2))


if __name__ == "__main__":
    main()
