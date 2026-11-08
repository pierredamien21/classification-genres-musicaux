"""Reproduce selected validation inputs and predictions without training or relabeling."""

import csv
import json
import os
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from analyze_validation_errors import load_inputs
from threadpoolctl import threadpool_limits

from music_genre.audio.processing import load_audio, segment_audio
from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.audio_diagnostic import diagnose_window
from music_genre.features.hpss import MusicalHpssViT
from music_genre.features.logmel import log_mel
from music_genre.features.vit import MusicalViT
from music_genre.models.linear import LinearProbabilityModel
from music_genre.models.vit_block import FrozenViTBlock

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_cache(name, baseline, labels):
    pointer = read_json(ROOT / f"data/features/latest_{name}.json")
    folder = (ROOT / pointer["folder"]).resolve()
    if (not folder.is_relative_to((ROOT / "data/features").resolve())
            or sha256_file(folder / "report.json") != pointer["report_sha256"]):
        raise ValueError("Invalid feature cache")
    report = read_json(folder / "report.json")
    provenance = report["provenance"]
    if provenance["manifest_sha256"] != baseline["manifest_sha256"] or provenance["labels"] != labels:
        raise ValueError("Cache cohort differs")
    for source, expected in provenance["source_sha256"].items():
        if sha256_file(ROOT / source) != expected:
            raise ValueError(f"Extraction source changed: {source}")
    for package, expected in provenance["versions"].items():
        if version(package) != expected:
            raise ValueError(f"Extraction version changed: {package}")
    for filename, key in (("index.csv", "index_sha256"), ("features.npy", "features_sha256")):
        if sha256_file(folder / filename) != report[key]:
            raise ValueError("Cached array or index changed")
    with (folder / "index.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    matrix = np.load(folder / "features.npy", allow_pickle=False, mmap_mode="r")
    index = {}
    for i, row in enumerate(rows):
        key = row["track_id"], int(row["window_id"])
        if int(row["row_index"]) != i or key in index:
            raise ValueError("Invalid cache row order")
        index[key] = row
    if len(matrix) != len(rows):
        raise ValueError("Cache size differs from index")
    return pointer, report, index, matrix


def plot_track(folder, item, audio, channels):
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "data/interim/matplotlib"))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(12, 10), layout="constrained")
    grid = fig.add_gridspec(4, 3, height_ratios=[0.7, 1, 1, 1])
    wave = fig.add_subplot(grid[0, :])
    # Plot min/max over short blocks so transients are not hidden by downsampling.
    width = 160
    blocks = audio.samples[:len(audio.samples) // width * width].reshape(-1, width)
    t = np.arange(len(blocks)) * width / audio.sample_rate
    wave.fill_between(t, blocks.min(axis=1), blocks.max(axis=1), color="#24656e", linewidth=0)
    for boundary in (0, 10, 20, 30):
        wave.axvline(boundary, color="#ac5132", linestyle="--", linewidth=1)
    wave.set(xlabel="Temps dans l’extrait (s)", ylabel="Amplitude", title="Signal mono 16 kHz ; limites des fenêtres")
    axes = []
    for window_id, values in enumerate(channels):
        for channel, label in enumerate(("Original", "Harmonique", "Percussif")):
            ax = fig.add_subplot(grid[window_id + 1, channel]); axes.append(ax)
            im = ax.imshow(values[channel], origin="lower", aspect="auto", cmap="magma", vmin=-80, vmax=20,
                           extent=(window_id * 10, (window_id + 1) * 10, 0, 128))
            ax.set(title=f"{label} · {window_id * 10}–{(window_id + 1) * 10} s", xlabel="Temps (s)", ylabel="Bande Mel")
    fig.colorbar(im, ax=axes, shrink=0.8, label="dB, référence fixe 1 ; affichage borné à [−80, 20]")
    fig.suptitle(f"Diagnostic {item['track_id']} — étiquette FMA : {item['truth']}\n"
                 "Spectrogrammes natifs avant redimensionnement ; aucune conclusion d’écoute")
    path = folder / f"{item['track_id']}.png"
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256_file(path)}


def main():
    settings = read_json(ROOT / "configs/audio_diagnostic.json")
    if settings["split"] != "validation" or settings["models"] != ["vit", "vit_block6", "vit_hpss"]:
        raise ValueError("Unsupported diagnostic protocol")
    review_pointer = read_json(ROOT / "reports/latest_validation_errors.json")
    review_path = (ROOT / review_pointer["report"]).resolve()
    if (not review_path.is_relative_to((ROOT / "reports").resolve())
            or sha256_file(review_path) != review_pointer["report_sha256"]):
        raise ValueError("Error review changed")
    review = read_json(review_path)
    packet = (ROOT / review["review_folder"]).resolve()
    if (not packet.is_relative_to((ROOT / "data/interim").resolve())
            or sha256_file(packet / "queue.json") != review["packet_sha256"]["queue.json"]):
        raise ValueError("Review queue changed")
    queue = read_json(packet / "queue.json")
    baseline, sources, labels, ids, _truth, saved_probabilities, runs, run_pointers = load_inputs()
    if baseline != review["baseline"] or run_pointers != review["runs"]:
        raise ValueError("Reviewed models or cohort changed")
    caches = {}
    for name in ("cnn_logmel", *settings["models"]):
        print(f"Verifying cache {name}", flush=True)
        caches[name] = load_cache(name, baseline, labels)
        if name in runs and (caches[name][0]["folder"] != runs[name]["feature_cache"]
                             or caches[name][0]["report_sha256"] != runs[name]["feature_report_sha256"]):
            raise ValueError("Cache differs from trained model inputs")
    config = caches["cnn_logmel"][1]["provenance"]["config"]
    hpss_config = caches["vit_hpss"][1]["provenance"]["config"]
    original = MusicalViT(caches["vit"][1]["provenance"]["config"], ROOT)
    hpss = MusicalHpssViT(hpss_config, ROOT)
    block6 = FrozenViTBlock(ROOT / hpss_config["checkpoint_directory"], 6)
    classifiers = {name: LinearProbabilityModel.load(ROOT / run_pointers[name]["folder"] / "model.npz")
                   for name in settings["models"]}
    for model in classifiers.values():
        if not np.array_equal(model.classes, np.arange(len(labels))):
            raise ValueError("Class order changed")
    for name in settings["models"]:
        cfg = caches[name][1]["provenance"]["config"]
        if {k: cfg[k] for k in config} != config or cfg["db_min"] != -80 or cfg["db_max"] != 20:
            raise ValueError("Audio preprocessing differs")
    folder = ROOT / "reports/figures" / ("audio-diagnostic-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    folder.mkdir(parents=True, exist_ok=False)
    results, plotted = [], set()
    with threadpool_limits(limits=1):
        for number, item in enumerate(queue, 1):
            key = item["track_id"]
            source = sources[key]
            if item["split"] != "validation" or source["split"] != "validation" or item["source_sha256"] != source["source_sha256"]:
                raise ValueError("Review track is not the selected validation source")
            path = (ROOT / "data/raw" / source["audio_path"]).resolve()
            if not path.is_relative_to((ROOT / "data/raw/fma_small").resolve()) or sha256_file(path) != source["source_sha256"]:
                raise ValueError("Audio source changed")
            audio = load_audio(path, sample_rate=config["sample_rate"])
            windows = segment_audio(audio, tuple(config["offsets_seconds"]), config["window_seconds"])
            if len(windows) != 3:
                raise ValueError("Expected three existing review windows")
            diagnostics, channels, embeddings = [], [], {name: [] for name in settings["models"]}
            embedding_errors = {name: [] for name in settings["models"]}
            for window_id, window in enumerate(windows):
                rows = {name: value[2][(key, window_id)] for name, value in caches.items()}
                ref = rows["cnn_logmel"]
                for row in rows.values():
                    if any(row[field] != ref[field] for field in ("start_seconds", "valid_seconds", "split", "artist_id", "label_index", "source_sha256")):
                        raise ValueError("Window identity differs between caches")
                    if row["source_sha256"] != source["source_sha256"] or row["split"] != "validation":
                        raise ValueError("Window source differs")
                mel = log_mel(window, config["logmel"])
                diag = diagnose_window(window, mel, caches["cnn_logmel"][3][int(ref["row_index"])], ref,
                                       settings["logmel_absolute_tolerance_db"])
                c = hpss.spectrograms(window)
                if not np.allclose(c[0], mel, rtol=0, atol=settings["logmel_absolute_tolerance_db"]):
                    raise ValueError("HPSS original channel differs from native input")
                channels.append(c)
                diag["channels_below_db_range_fraction"] = [float(np.mean(x < -80)) for x in c]
                diag["channels_above_db_range_fraction"] = [float(np.mean(x > 20)) for x in c]
                diagnostics.append(diag)
                values = {"vit": original.encoder.extract(mel, -80, 20),
                          "vit_block6": block6.extract(mel, -80, 20),
                          "vit_hpss": hpss.encoder.extract(c, -80, 20)}
                for name, x in values.items():
                    error = float(np.max(np.abs(x - caches[name][3][int(rows[name]["row_index"]) ])))
                    if not np.isfinite(error) or error > settings["embedding_absolute_tolerance"]:
                        raise ValueError(f"Embedding differs from cache: {name}/{key}/{window_id}: {error}")
                    embeddings[name].append(x)
                    embedding_errors[name].append(error)
            predictions = {}
            for name, values in embeddings.items():
                probabilities = classifiers[name].predict_proba(np.stack(values))
                mean = probabilities.mean(axis=0)
                error = float(np.max(np.abs(mean - saved_probabilities[name][ids.index(key)])))
                if error > settings["probability_absolute_tolerance"]:
                    raise ValueError(f"Track probabilities differ: {name}/{key}: {error}")
                predictions[name] = {"window_labels": [labels[int(p.argmax())] for p in probabilities],
                                     "window_true_probabilities": probabilities[:, int(source["label_index"])].tolist(),
                                     "track_label": labels[int(mean.argmax())], "max_probability_error": error,
                                     "max_embedding_error": max(embedding_errors[name])}
            record = {"track_id": key, "stratum": item["stratum"], "true_label": source["genre"],
                      "source_sha256": source["source_sha256"], "duration_seconds": len(audio.samples) / audio.sample_rate,
                      "source_sample_rate": audio.source_sample_rate, "source_channels": audio.source_channels,
                      "windows": diagnostics, "predictions": predictions}
            if item["stratum"] not in plotted:
                record["figure"] = plot_track(folder, item, audio, channels)
                plotted.add(item["stratum"])
            results.append(record)
            print(f"Verified {number}/{len(queue)}: {key}", flush=True)
    output = {"schema_version": 1, "status": "complete", "tracks": results,
              "review": review_pointer, "review_queue_sha256": review["packet_sha256"]["queue.json"],
              "settings": settings, "config_sha256": sha256_file(ROOT / "configs/audio_diagnostic.json"),
              "caches": {name: value[0] for name, value in caches.items()}, "runs": run_pointers,
              "test_audio_loaded": False, "test_evaluated": False, "semantic_listening_performed": False,
              "source_sha256": {name: sha256_file(ROOT / name) for name in (
                  "scripts/diagnose_review_audio.py", "src/music_genre/evaluation/audio_diagnostic.py")}}
    write_json(ROOT / "reports/AUDIO_DIAGNOSTIC.json", output)
    write_json(ROOT / "reports/latest_audio_diagnostic.json", {
        "report": "reports/AUDIO_DIAGNOSTIC.json", "report_sha256": sha256_file(ROOT / "reports/AUDIO_DIAGNOSTIC.json")})
    lines = ["# Diagnostic technique des 19 extraits de revue", "",
             "Recalcul depuis les MP3 originaux sur validation uniquement ; aucun entraînement ni changement de label.", "",
             "## Correspondance avec les entrées et résultats enregistrés", "",
             "| Piste | Genre FMA | Padding maximal (ms) | Erreur log-Mel max (dB) | Erreur embedding max | Erreur probabilité max |",
             "|---|---|---:|---:|---:|---:|"]
    for r in results:
        lines.append(f"| {r['track_id']} | {r['true_label']} | {1000 * max(w['padded_seconds'] for w in r['windows']):.2f} | "
                     f"{max(w['max_cache_error_db'] for w in r['windows']):.3g} | "
                     f"{max(p['max_embedding_error'] for p in r['predictions'].values()):.3g} | "
                     f"{max(p['max_probability_error'] for p in r['predictions'].values()):.3g} |")
    lines += ["", "## Prédictions HPSS selon la fenêtre", "",
              "| Piste | Genre FMA | 0–10 s | 10–20 s | 20–30 s | Agrégation |", "|---|---|---|---|---|---|"]
    for r in results:
        p = r["predictions"]["vit_hpss"]
        lines.append(f"| {r['track_id']} | {r['true_label']} | " + " | ".join(p["window_labels"]) + f" | {p['track_label']} |")
    lines += ["", "## Figures représentatives", "",
              "Une piste par catégorie, première dans la sélection préexistante. Ces figures révèlent les labels : les consulter après l’écoute indépendante.", ""]
    for r in results:
        if "figure" in r:
            lines.append(f"- [{r['track_id']} — {r['stratum']}](../{r['figure']['path']})")
    lines += ["", "## Portée", "",
              ("Les fenêtres, spectrogrammes natifs, embeddings des trois ViT et probabilités par morceau ont été recalculés "
               "et comparés aux caches et prédictions enregistrés. Le canal original HPSS est comparé au log-Mel natif. "
               "Les empreintes des sources, versions, modèles, fichiers audio et caches sont contrôlées."), "",
              ("Cette reproduction détecte un écart avec le pipeline enregistré ; elle ne prouve pas que ses choix de mono, "
               "16 kHz, fenêtres, redimensionnement ou préentraînement image sont optimaux. Un défaut partagé par extraction "
               "et recalcul peut subsister. Les trois CNN ne sont pas réexécutés : leur entrée log-Mel commune est contrôlée."), "",
              ("RMS calculé sur les seuls échantillons valides, sans padding ; les fractions hors plage dB quantifient "
               "l’écrêtage de la représentation, pas une saturation audible. Aucun genre ni instrument n’est identifié par écoute ici."), "",
              "[Mesures par fenêtre, toutes les prédictions et empreintes](AUDIO_DIAGNOSTIC.json).", "",
              "Commande : `.\\.venv\\Scripts\\python.exe scripts/diagnose_review_audio.py`.", ""]
    (ROOT / "reports/DIAGNOSTIC_AUDIO.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Complete: {len(results)} tracks, {sum(len(r['windows']) for r in results)} windows; test excluded", flush=True)


if __name__ == "__main__":
    main()
