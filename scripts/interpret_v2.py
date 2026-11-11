"""V2 phase 6: descriptive interpretation of the single V2 evaluation, and the V2 listening packet.

  python scripts/interpret_v2.py

Reads the archived final evaluation (reports/v2/final_evaluation), the development out-of-fold
predictions of three CV runs and the FMA metadata. Writes reports/v2/interpretation.json and a new
listening packet data/interim/error-review-v2-<timestamp>/ (audio embedded, labels hidden until
opened). Nothing here feeds a model, a threshold or a decision: the V2 test is spent.
"""

import ast
import base64
import hashlib
import html
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_features import selected_rows

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.calibration import temperature_scale

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "reports/v2/final_evaluation"
CV_RUNS = {"maest_layer12": "runs/v2-cv/dev_maest-layer12-20261109T212736039178Z",
           "vit_block6_hpss": "runs/v2-cv/dev_vit_block6_hpss-20261109T211716619642Z",
           "vit": "runs/v2-cv/dev_vit-20261109T212849167407Z"}
SEED, REPLICATES, PER_STRATUM = 20261111, 2000, 3
LABELS = json.loads((ROOT / "configs/labels.json").read_text())["labels"]
K = range(len(LABELS))


def f1(truth, predicted, weight=None):
    return float(f1_score(truth, predicted, labels=K, average="macro", zero_division=0, sample_weight=weight))


def probabilities(frame):
    return frame[[f"p_{label}" for label in LABELS]].to_numpy()


def load():
    report = json.loads((FINAL / "report.json").read_text(encoding="utf-8"))
    if sha256_file(FINAL / "predictions.csv") != report["predictions_sha256"]:
        raise ValueError("Archived predictions changed")
    test = pd.read_csv(FINAL / "predictions.csv", dtype={"track_id": str, "artist_id": str})
    oof = {}
    for system, run in CV_RUNS.items():
        result = json.loads((ROOT / run / "report.json").read_text(encoding="utf-8"))
        if sha256_file(ROOT / run / "oof_tracks.csv") != result["oof_tracks_sha256"]:
            raise ValueError(f"Out-of-fold file changed: {run}")
        oof[system] = pd.read_csv(ROOT / run / "oof_tracks.csv", dtype={"track_id": str}).set_index("track_id")
    folds = pd.read_csv(ROOT / "reports/v2/dev_folds.csv", dtype=str).set_index("track_id")
    _, manifest = selected_rows(LABELS, ("train", "validation", "test"))
    folds["v1_split"] = pd.Series({r["track_id"]: r["split"] for r in manifest})
    metadata = pd.read_csv(ROOT / "data/raw/fma_metadata/tracks.csv", header=[0, 1], index_col=0, low_memory=False)
    metadata.index = [f"{i:06d}" for i in metadata.index]
    return report, test, oof, folds, metadata


def by_genre(test, oof):
    result = {}
    for system, frame in oof.items():
        t = test[(test.role == "test") & (test.system == system)]
        dev = f1_score(frame.label_index, probabilities(frame).argmax(1), labels=K, average=None, zero_division=0)
        held = f1_score(t.genre.map(LABELS.index), probabilities(t).argmax(1), labels=K, average=None, zero_division=0)
        result[system] = {label: {"development_oof": float(a), "v2_test": float(b), "change": float(b - a)}
                          for label, a, b in zip(LABELS, dev, held, strict=True)}
    frame, t = oof["maest_layer12"], test[(test.role == "test") & (test.system == "maest_layer12")]

    def rates(truth, predicted):
        matrix = confusion_matrix(truth, predicted, labels=K)
        return matrix / matrix.sum(axis=1, keepdims=True)

    shift = rates(t.genre.map(LABELS.index), probabilities(t).argmax(1)) - rates(
        frame.label_index, probabilities(frame).argmax(1))
    largest = sorted(((shift[i, j], LABELS[i], LABELS[j]) for i in K for j in K if i != j), reverse=True)[:6]
    result["maest_confusions_that_grew"] = [{"true": a, "predicted": b, "rate_change": float(v)}
                                            for v, a, b in largest]
    return result


def artists(test, oof, folds):
    result = {}
    for system, frame in oof.items():
        artist = folds.loc[frame.index, "artist_id"]
        weight = 1 / artist.map(artist.value_counts()).to_numpy()
        t = test[(test.role == "test") & (test.system == system)]
        result[system] = {"development_track_weighted": f1(frame.label_index, probabilities(frame).argmax(1)),
                          "development_artist_weighted": f1(frame.label_index, probabilities(frame).argmax(1), weight),
                          "v2_test_one_track_per_artist": f1(t.genre.map(LABELS.index), probabilities(t).argmax(1))}
    frame = oof["maest_layer12"]
    artist = folds.loc[frame.index, "artist_id"]
    size, correct = artist.map(artist.value_counts()).to_numpy(), probabilities(frame).argmax(1) == frame.label_index
    result["maest_accuracy_by_artist_tracks_in_development"] = {
        f"{lo}-{hi}": {"tracks": int(((size >= lo) & (size <= hi)).sum()),
                       "accuracy": float(correct[(size >= lo) & (size <= hi)].mean())}
        for lo, hi in ((1, 1), (2, 3), (4, 9), (10, 10 ** 6))}
    return result


def v1_gap(oof, folds):
    """Out-of-fold F1 on the V1 validation and V1 test tracks; artist-cluster bootstrap of the gap."""
    v1_val = json.loads((ROOT / "reports/FINAL_VALIDATION_RECOUNT.json").read_text(encoding="utf-8"))["systems"]
    v1_test = json.loads((ROOT / "reports/FINAL_TEST_RECOUNT.json").read_text(encoding="utf-8"))["systems"]
    result = {}
    for system, frame in oof.items():
        split = folds.loc[frame.index, "v1_split"].to_numpy()
        truth, predicted = frame.label_index.to_numpy(), probabilities(frame).argmax(1)
        artist = folds.loc[frame.index, "artist_id"].to_numpy()
        inside = np.isin(split, ("validation", "test"))
        groups = {a: np.flatnonzero(inside & (artist == a)) for a in np.unique(artist[inside])}
        names = np.array(list(groups))
        rng = np.random.Generator(np.random.PCG64(SEED))
        draws = []
        for _ in range(REPLICATES):
            rows = np.concatenate([groups[a] for a in rng.choice(names, len(names))])
            val, tst = rows[split[rows] == "validation"], rows[split[rows] == "test"]
            draws.append(f1(truth[val], predicted[val]) - f1(truth[tst], predicted[tst]))
        val, tst = split == "validation", split == "test"
        result[system] = {
            "oof_v1_validation": f1(truth[val], predicted[val]), "oof_v1_test": f1(truth[tst], predicted[tst]),
            "oof_gap": f1(truth[val], predicted[val]) - f1(truth[tst], predicted[tst]),
            "oof_gap_interval_95": [float(q) for q in np.quantile(draws, [0.025, 0.975])],
            "v1_official_validation": v1_val.get(system, {}).get("macro_f1"),
            "v1_official_test": v1_test.get(system, {}).get("macro_f1"),
        }
    result["method"] = (f"V2 CV heads never selected on the V1 validation; artist-cluster bootstrap over the V1 "
                        f"validation and test artists, {REPLICATES} draws, seed {SEED}, percentile interval")
    return result


def covariates(test, oof, metadata):
    frame = oof["maest_layer12"]
    t = test[(test.role == "test") & (test.system == "maest_layer12")].set_index("track_id")
    dev_ok = pd.Series(probabilities(frame).argmax(1) == frame.label_index.to_numpy(), index=frame.index)
    test_ok = pd.Series(probabilities(t).argmax(1) == t.genre.map(LABELS.index).to_numpy(), index=t.index)

    def table(ids):
        rows = metadata.loc[ids]
        artist_counts = metadata[("artist", "id")].value_counts()
        return pd.DataFrame({
            "listens": rows[("track", "listens")].to_numpy(),
            "bit_rate": rows[("track", "bit_rate")].to_numpy(),
            "year_created": pd.to_datetime(rows[("track", "date_created")]).dt.year.to_numpy(),
            "genres_all": rows[("track", "genres_all")].map(lambda s: len(ast.literal_eval(s))).to_numpy(),
            "artist_tracks_in_fma": rows[("artist", "id")].map(artist_counts).to_numpy()}, index=ids)

    dev, held = table(frame.index), table(t.index)
    result = {"accuracy": {"development_oof": float(dev_ok.mean()), "v2_test": float(test_ok.mean())},
              "method": "development accuracy by quartile bin, reweighted to the V2 test bin shares"}
    for column in dev.columns:
        edges = np.unique(np.quantile(pd.concat([dev[column], held[column]]), [0, 0.25, 0.5, 0.75, 1]))
        dev_bins = pd.cut(dev[column], edges, include_lowest=True)
        test_bins = pd.cut(held[column], edges, include_lowest=True)
        dev_accuracy = dev_ok.groupby(dev_bins.to_numpy(), observed=False).mean()
        test_share = test_bins.value_counts(normalize=True).sort_index()
        result[column] = {
            "median_development": float(dev[column].median()), "median_v2_test": float(held[column].median()),
            "bins": [str(b) for b in dev_accuracy.index],
            "development_accuracy": dev_accuracy.round(4).tolist(),
            "v2_test_accuracy": test_ok.groupby(test_bins.to_numpy(), observed=False).mean().round(4).tolist(),
            "v2_test_share": test_share.round(4).tolist(),
            "development_accuracy_reweighted_to_test": float((test_share * dev_accuracy).sum())}
    return result


def replication(report):
    v1_val = json.loads((ROOT / "reports/FINAL_VALIDATION_RECOUNT.json").read_text(encoding="utf-8"))["systems"]
    v1_test = json.loads((ROOT / "reports/FINAL_TEST_RECOUNT.json").read_text(encoding="utf-8"))["systems"]
    rows = {s: {"v1_validation": v1_val[s]["macro_f1"], "v1_test": v1_test[s]["macro_f1"],
                "v2_test": report["systems"][s]["macro_f1"]["point"],
                "v2_test_interval_95": [report["systems"][s]["macro_f1"]["low"], report["systems"][s]["macro_f1"]["high"]]}
            for s in ("vit_block6_hpss", "vit", "vit_block6", "cnn20_seed42", "cnn20_seed43", "cnn20_seed44")}
    h1 = report["hypotheses"]["H1"]
    return {"systems": rows, "H1": {
        "v1_test": {"point": 0.0394, "interval_95": [0.0054, 0.0715], "source": "reports/EVALUATION_FINALE.md"},
        "v2_test": {"point": h1["point"], "interval_95": [h1["low"], h1["high"]], "p_value": h1["p_value"]}},
        "reading": "same frozen V1 exports on a new, independent test: a replication, not a gain on the V1 test"}


def listening_packet(report, test, metadata):
    """Seeded selection of V2 test and out-of-domain tracks for listening; one track per artist overall."""
    calibration = json.loads((ROOT / json.loads((ROOT / "configs/evaluation_v2.json").read_text())["calibration"][
        "report"]).read_text(encoding="utf-8"))["final"]
    cohorts = pd.read_csv(ROOT / "reports/v2/final_cohorts.csv", dtype=str).set_index("track_id")
    maest = test[test.system == "maest_layer12"].set_index("track_id")
    hpss = test[test.system == "vit_block6_hpss"].set_index("track_id")
    scaled = temperature_scale(probabilities(maest), calibration["temperature"])
    maest = maest.assign(predicted=[LABELS[i] for i in scaled.argmax(1)], confidence=scaled.max(1))
    in_test = maest.role == "test"
    wrong = in_test & (maest.predicted != maest.genre)
    strata = {
        "confident_error": maest[wrong & (maest.confidence >= maest[wrong].confidence.quantile(0.75))].index,
        "pop_error": maest[wrong & (maest.genre == "Pop")].index,
        "instrumental_error": maest[wrong & (maest.genre == "Instrumental")].index,
        "false_experimental": maest[wrong & (maest.predicted == "Experimental")].index,
        "correct_control": maest[in_test & ~wrong].index,
        "accepted_out_of_domain": maest[(maest.role == "ood") & (maest.confidence >= calibration[
            "threshold_top_probability"])].index,
    }
    selected, used_artists = [], set()
    for name in sorted(strata, key=lambda s: (len(strata[s]), s)):
        order = sorted(strata[name], key=lambda t: hashlib.sha256(f"{SEED}:{name}:{t}".encode()).hexdigest())
        chosen = [t for t in order if cohorts.loc[t, "artist_id"] not in used_artists][:PER_STRATUM]
        for t in chosen:
            used_artists.add(cohorts.loc[t, "artist_id"])
            selected.append((name, t))
    folder = ROOT / "data/interim" / f"error-review-v2-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    folder.mkdir(parents=True)
    queue, cards = [], []
    for number, (stratum, t) in enumerate(selected, 1):
        source, meta = cohorts.loc[t], metadata.loc[t]
        path = (ROOT / "data/raw" / source["audio_path"]).resolve()
        if not path.is_relative_to((ROOT / "data/raw").resolve()) or sha256_file(path) != source["sha256"]:
            raise ValueError(f"Listening audio changed: {t}")
        shown = {"MAEST bloc 12 (candidat V2, calibré)": (maest.loc[t, "predicted"], maest.loc[t, "confidence"]),
                 "ViT bloc 6 + HPSS (candidat V1)": (LABELS[int(probabilities(hpss.loc[[t]]).argmax())],
                                                     float(probabilities(hpss.loc[[t]]).max()))}
        queue.append({"stratum": stratum, "track_id": t, "role": source["role"], "label": source["genre"],
                      "artist_id": source["artist_id"], "audio_path": path.relative_to(ROOT).as_posix(),
                      "source_sha256": source["sha256"], "license": meta[("track", "license")],
                      "predictions": {k: {"label": v[0], "probability": float(v[1])} for k, v in shown.items()},
                      "review_status": "pending", "observations": ""})
        rows = "".join(f"<tr><td>{html.escape(k)}</td><td>{html.escape(v[0])}</td><td>{v[1]:.3f}</td></tr>"
                       for k, v in shown.items())
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        kind = "Étiquette FMA" if source["role"] == "test" else "Genre FMA hors des huit classes"
        cards.append(f'''<article class="case" data-track="{t}"><h2>Extrait {number:02d}</h2>
<audio controls preload="none" src="data:audio/mpeg;base64,{encoded}"></audio>
<label>Observations d’écoute, genres plausibles et incertitudes<textarea aria-label="Observations extrait {number:02d}"></textarea></label>
<label><input type="checkbox"> Écoute et revue réalisées</label>
<details><summary>Afficher l’étiquette, les modèles et les métadonnées</summary>
<p><strong>{html.escape(str(meta[("track", "title")]))}</strong> — {html.escape(str(meta[("artist", "name")]))} · piste {t} · artiste {source["artist_id"]}<br>
{kind} : <strong>{html.escape(source["genre"])}</strong><br>Sélection : {html.escape(stratum)}</p>
<table><thead><tr><th>Modèle</th><th>Prédiction</th><th>Probabilité maximale</th></tr></thead><tbody>{rows}</tbody></table>
<p class="subtle">Une probabilité élevée ne prouve pas une erreur d’étiquette.<br>
Genres FMA détaillés (identifiants) : {html.escape(str(meta[("track", "genres_all")]))}.<br>
Licence : {html.escape(str(meta[("track", "license")]))}. SHA-256 : <code>{source["sha256"]}</code>.</p></details></article>''')
    page = (ROOT / "templates/error_review.html").read_text(encoding="utf-8")
    page = (page.replace("Revue audio — erreurs de validation", "Revue audio — erreurs du test V2")
            .replace("validation uniquement", "test V2 et hors domaine, après l’évaluation unique")
            .replace("pour examiner Pop et Hip-Hop", "parmi les erreurs du candidat V2, des contrôles corrects et "
                     "des morceaux hors domaine acceptés")
            .replace("__PACKET__", folder.name).replace("__COUNT__", str(len(queue))).replace("__CARDS__", "\n".join(cards)))
    (folder / "index.html").write_text(page, encoding="utf-8")
    write_json(folder / "queue.json", queue)
    write_json(folder / "annotations_template.json", {"schema_version": 1, "packet": folder.name, "annotations": [
        {"track_id": q["track_id"], "status": "pending", "observations": ""} for q in queue]})
    return {"folder": folder.relative_to(ROOT).as_posix(), "seed": SEED, "per_stratum": PER_STRATUM,
            "available": {k: len(v) for k, v in strata.items()},
            "selected": {k: sum(s == k for s, _ in selected) for k in strata},
            "files_sha256": {n: sha256_file(folder / n) for n in ("index.html", "queue.json", "annotations_template.json")},
            "rule": "rare strata first; SHA256(seed:stratum:track_id); one track per artist overall; listen before opening labels"}


def main():
    report, test, oof, folds, metadata = load()
    result = {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(),
        "final_report_sha256": sha256_file(FINAL / "report.json"),
        "cv_runs": {s: {"run": r, "report_sha256": sha256_file(ROOT / r / "report.json")} for s, r in CV_RUNS.items()},
        "by_genre": by_genre(test, oof), "artists": artists(test, oof, folds), "v1_gap": v1_gap(oof, folds),
        "covariates_maest": covariates(test, oof, metadata), "replication": replication(report),
        "listening": listening_packet(report, test, metadata),
        "source_sha256": sha256_file(Path(__file__)), "decisions_changed": False,
    }
    write_json(ROOT / "reports/v2/interpretation.json", result)
    print(json.dumps({k: result[k] for k in ("artists", "v1_gap")}, indent=1))
    print(json.dumps(result["listening"]["selected"]), result["listening"]["folder"])


if __name__ == "__main__":
    main()
