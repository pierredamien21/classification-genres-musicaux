"""V2 phase 5: the single inference on the V2 test and out-of-domain cohorts (configs/evaluation_v2.json).

  python scripts/evaluate_v2.py --rehearsal 3 --workers 7   # development dry run, no V2 test audio
  python scripts/evaluate_v2.py --lock                      # writes reports/v2/lock3.json
  python scripts/evaluate_v2.py --final --workers 7         # once, after lock 3

The rehearsal runs the whole pipeline on development tracks of fold 0 that are also V1 validation
tracks, and checks every system against predictions computed before (development caches, CNN
fold-0 predictions, V1 validation predictions). --lock pins systems, weights, code, configs and the
environment. --final refuses to run if any pinned file or package changed, or if the evaluation
already produced its report; an interrupted run resumes track by track under the same lock.
"""

import argparse
import csv
import json
import platform
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path

import numpy as np
from sklearn.metrics import f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_v2 import load_cache
from extract_features import selected_rows

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.calibration import expected_calibration_error, temperature_scale
from music_genre.evaluation.final_v2 import bca, fast_macro_f1, holm, ood_summary

ROOT = Path(__file__).resolve().parents[1]
CONFIG = "configs/evaluation_v2.json"
LOCK = "reports/v2/lock3.json"
FINAL = "runs/v2-final-evaluation"
WORKER = None


def settings():
    config = json.loads((ROOT / CONFIG).read_text(encoding="utf-8"))
    labels = json.loads((ROOT / "configs/labels.json").read_text())["labels"]
    pointers = {name: json.loads((ROOT / s["pointer"]).read_text()) for name, s in config["v2_systems"].items()}
    return config, labels, pointers


def build_predictors(config, pointers):
    from music_genre.inference.predict import load_predictor
    from music_genre.inference.predict_v2 import CnnV2Predictor, MaestPredictor

    predictors = {name: load_predictor(ROOT, name) for name in config["v1_systems"]}
    for name, system in config["v2_systems"].items():
        kind = MaestPredictor if system["kind"] == "maest_final_head" else CnnV2Predictor
        predictors[name] = kind(ROOT, pointers[name]["folder"], pointers[name]["report_sha256"])
    return predictors


def initialize(keep_features):
    global WORKER
    config, _, pointers = settings()
    WORKER = (build_predictors(config, pointers), keep_features)


def predict_track(task):
    row, folder = task
    target = Path(folder) / "tracks" / f"{row['track_id']}.npz"
    if not target.exists():
        path = (ROOT / "data/raw" / row["audio_path"]).resolve()
        if not path.is_relative_to((ROOT / "data/raw").resolve()) or sha256_file(path) != row["sha256"]:
            raise ValueError(f"Audio changed since the lock: {row['track_id']}")
        predictors, keep_features = WORKER
        arrays = {}
        for name, predictor in predictors.items():
            if hasattr(predictor, "predict_file"):  # V1 export
                arrays[f"p__{name}"] = np.array([w["scores"] for w in predictor.predict_file(path)["windows"]])
            else:
                arrays[f"p__{name}"], x = predictor.predict(path)
                if keep_features:
                    arrays[f"x__{name}"] = x
        temporary = target.with_suffix(".partial.npz")
        np.savez(temporary, source_sha256=row["sha256"], **arrays)
        temporary.replace(target)
    with np.load(target, allow_pickle=False) as saved:
        if str(saved["source_sha256"]) != row["sha256"]:
            raise ValueError("Stale track predictions")
        return row["track_id"], {k: saved[k] for k in saved.files if k != "source_sha256"}


def rehearsal_rows(labels, per_genre):
    """Development tracks with saved reference predictions: fold 0 (V2 CNN) and V1 validation (V1 systems).

    Tracks in both come first, then fold 0, then V1 validation; one track per artist.
    """
    with (ROOT / "reports/v2/dev_folds.csv").open(encoding="utf-8", newline="") as stream:
        folds = {r["track_id"]: r for r in csv.DictReader(stream)}
    _, manifest = selected_rows(labels, ("train", "validation", "test"))
    by_genre, artists = {label: [] for label in labels}, set()

    def priority(r):
        return (folds[r["track_id"]]["fold"] != "0") * 2 + (r["split"] != "validation"), r["track_id"]

    for r in sorted(manifest, key=priority):
        fold = folds[r["track_id"]]
        if (fold["fold"] == "0" or r["split"] == "validation") and fold["artist_id"] not in artists:
            artists.add(fold["artist_id"])
            by_genre[r["genre"]].append({"genre": r["genre"], "track_id": r["track_id"], "artist_id": fold["artist_id"],
                                         "audio_path": r["audio_path"], "sha256": r["source_sha256"]})
    rows = []
    for i, label in enumerate(labels):
        chosen = by_genre[label]
        rows += [{**r, "role": "test"} for r in chosen[:per_genre]]
        rows += [{**r, "role": "ood"} for r in chosen[per_genre:2 * per_genre]]
        rows += [{**r, "role": "gospel_probe"} for r in chosen[2 * per_genre:2 * per_genre + (i < 2)]]
    return rows


def final_rows():
    config, _, _ = settings()
    with (ROOT / config["cohorts"]).open(encoding="utf-8", newline="") as stream:
        return [{k: r[k] for k in ("role", "genre", "track_id", "artist_id", "audio_path", "sha256")}
                for r in csv.DictReader(stream)]


def environment():
    return {"python": platform.python_version(), "implementation": platform.python_implementation(),
            "machine": platform.machine(),
            "packages": sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions())}


def lock_files(config, pointers, rehearsal):
    files = {p.relative_to(ROOT).as_posix() for p in (ROOT / "src/music_genre").rglob("*.py")}
    files |= {f"scripts/{s}.py" for s in ("evaluate_v2", "fit_final_v2", "cv_v2", "extract_features", "train_cnn",
                                           "calibrate_v2", "robustness_v2")}
    files |= {f"configs/{c}.json" for c in ("protocol_v2", "evaluation_v2", "labels", "maest", "cnn_v2", "cv_v2",
                                             "calibration_v2", "degradations_v2")}
    files |= {"docs/PROTOCOLE_V2.md", "pyproject.toml", config["cohorts"], config["calibration"]["report"],
              config["decoder_sensitivity"]["csv"], "reports/v2/lock2.json", "reports/v2/candidate.json",
              "reports/v2/phase4.json", "reports/v2/dev_folds.csv", "reports/PREPARATION_EVALUATION_FINALE.json",
              "reports/latest_final_evaluation_lock.json", rehearsal}
    files |= {p.relative_to(ROOT).as_posix() for p in ROOT.glob("requirements/*.txt")}
    files |= {p.relative_to(ROOT).as_posix() for d in ("maest", "vit_tiny")
              for p in (ROOT / "artifacts/pretrained" / d).iterdir() if p.is_file()}
    for name, system in config["v2_systems"].items():
        folder = pointers[name]["folder"]
        model = "model.npz" if system["kind"] == "maest_final_head" else "model.safetensors"
        files |= {system["pointer"], f"{folder}/report.json", f"{folder}/{model}"}
        report = json.loads((ROOT / folder / "report.json").read_text(encoding="utf-8"))
        files.add(f"{report['feature_cache']}/report.json")
    preparation = json.loads((ROOT / "reports/PREPARATION_EVALUATION_FINALE.json").read_text(encoding="utf-8"))
    for name in config["v1_systems"]:
        pin = preparation["runs"][name]
        files |= {pin["model_path"], f"{pin['run']['folder']}/report.json", f"{pin['feature_cache']}/report.json"}
    return sorted(files)


def verify_lock():
    lock = json.loads((ROOT / LOCK).read_text(encoding="utf-8"))
    if lock["status"] != "locked" or lock["test_evaluated"]:
        raise ValueError("Lock 3 is not in force")
    changed = [p for p, h in lock["files_sha256"].items() if not (ROOT / p).is_file() or sha256_file(ROOT / p) != h]
    if changed:
        raise ValueError(f"{len(changed)} pinned files changed since lock 3, e.g. {changed[:5]}")
    if environment() != lock["environment"]:
        raise ValueError("Python environment changed since lock 3")
    return sha256_file(ROOT / LOCK)


def statistics(rows, probs, config, labels, temperature, threshold, excluded):
    roles = np.array([r["role"] for r in rows])
    test = np.flatnonzero(roles == "test")
    if len({rows[i]["artist_id"] for i in test}) != len(test):
        raise ValueError("The test must have one track per artist")
    truth = np.array([labels.index(rows[i]["genre"]) for i in test])
    predicted = {s: p[test].argmax(axis=1) for s, p in probs.items()}
    boot = config["bootstrap"]

    def interval(statistic):
        return bca(statistic, len(test), replicates=boot["replicates"], seed=boot["seed"], level=boot["level"])

    def f1(system, i):
        return fast_macro_f1(truth[i], predicted[system][i], len(labels))

    systems = {}
    for s in probs:
        systems[s] = {
            "macro_f1": interval(lambda i, s=s: f1(s, i)),
            "accuracy": float(np.mean(predicted[s] == truth)),
            "per_genre_f1": dict(zip(labels, f1_score(truth, predicted[s], labels=range(len(labels)), average=None,
                                                      zero_division=0).tolist(), strict=True)),
            "confusion_matrix": np.bincount(truth * len(labels) + predicted[s],
                                            minlength=len(labels) ** 2).reshape(len(labels), -1).tolist(),
        }
    hypotheses = {}
    for name, h in config["hypotheses"].items():
        hypotheses[name] = {**h, **interval(
            lambda i, h=h: f1(h["system"], i) - np.mean([f1(b, i) for b in h["minus_mean_of"]]))}
    hypotheses["H1"]["rejected"] = hypotheses["H1"]["p_value"] < config["alpha"]
    secondary = holm({n: hypotheses[n]["p_value"] for n, h in config["hypotheses"].items()
                      if h["role"] == "secondary"}, config["alpha"])
    for name, result in secondary.items():
        hypotheses[name].update(result)
    candidate = probs[config["candidate"]]
    calibrated = temperature_scale(candidate, temperature)
    one_hot = np.eye(len(labels))[truth]
    calibration = {f"{kind}": {"ece": expected_calibration_error(p[test], truth, config["calibration"]["ece_bins"]),
                               "brier": float(np.mean(np.sum((p[test] - one_hot) ** 2, axis=1)))}
                   for kind, p in (("uncalibrated", candidate), ("calibrated", calibrated))}
    confidence = calibrated.max(axis=1)
    ood = np.flatnonzero(roles == "ood")
    gospel = np.flatnonzero(roles == "gospel_probe")
    out_of_domain = {
        "score": "maximum temperature-scaled probability of the candidate", "temperature": temperature,
        "threshold": threshold, "ood": ood_summary(confidence[test], confidence[ood], threshold),
        "gospel_probe": ood_summary(confidence[test], confidence[gospel], threshold),
        "accepted_share_by_ood_genre": {g: float(np.mean(confidence[[i for i in ood if rows[i]["genre"] == g]] >= threshold))
                                        for g in sorted({rows[i]["genre"] for i in ood})},
    }
    kept = np.array([rows[i]["track_id"] not in excluded for i in test])
    sensitivity = {"excluded_tracks": int((~kept).sum()), "macro_f1": {
        s: fast_macro_f1(truth[kept], predicted[s][kept], len(labels)) for s in probs}}
    h1 = config["hypotheses"]["H1"]
    sensitivity["H1_point"] = sensitivity["macro_f1"][h1["system"]] - np.mean(
        [sensitivity["macro_f1"][b] for b in h1["minus_mean_of"]])
    return {"test_tracks": len(test), "ood_tracks": len(ood), "gospel_tracks": len(gospel),
            "test_counts": dict(Counter(rows[i]["genre"] for i in test)), "systems": systems,
            "hypotheses": hypotheses, "calibration_on_test": calibration, "out_of_domain": out_of_domain,
            "decoder_sensitivity": sensitivity}


def rehearsal_checks(rows, saved, config, pointers, labels):
    """Each system against predictions made before: development caches and saved validation files."""
    ids = [r["track_id"] for r in rows]
    checks = {}
    _, report, index, x = load_cache("dev_maest")
    final = json.loads((ROOT / pointers["maest_layer12"]["folder"] / "report.json").read_text(encoding="utf-8"))
    from music_genre.models.linear import LinearProbabilityModel
    head = LinearProbabilityModel.load(ROOT / pointers["maest_layer12"]["folder"] / "model.npz")
    columns = [i for i, f in enumerate(report["feature_families"]) if f == final["family"]]
    positions = {}
    for position, row in enumerate(index):
        positions.setdefault(row["track_id"], []).append(position)
    feature_error = probability_error = 0.0
    for t in ids:
        cached = x[positions[t]][:, columns]
        feature_error = max(feature_error, float(np.max(np.abs(saved[t]["x__maest_layer12"] - cached))))
        probability_error = max(probability_error, float(np.max(np.abs(
            saved[t]["p__maest_layer12"] - head.predict_proba(cached)))))
    checks["maest_layer12"] = {"reference": "development cache and final head", "max_feature_error": feature_error,
                               "max_window_probability_error": probability_error}
    preparation = json.loads((ROOT / "reports/PREPARATION_EVALUATION_FINALE.json").read_text(encoding="utf-8"))
    folders = {n: preparation["runs"][n]["run"]["folder"] for n in config["v1_systems"]}
    folders |= {n: pointers[n]["folder"] for n, s in config["v2_systems"].items() if s["kind"] == "cnn_v2"}
    for name, folder in folders.items():
        with (ROOT / folder / "validation_predictions.csv").open(encoding="utf-8", newline="") as stream:
            reference = {r["track_id"]: np.array([float(r[f"p_{label}"]) for label in labels])
                         for r in csv.DictReader(stream)}
        compared = [t for t in ids if t in reference]
        if len(compared) < 8:
            raise ValueError(f"Too few rehearsal tracks with a reference for {name}")
        checks[name] = {"reference": f"{folder}/validation_predictions.csv", "tracks_compared": len(compared),
                        "max_track_probability_error": max(float(np.max(np.abs(
                            saved[t][f"p__{name}"].mean(axis=0) - reference[t]))) for t in compared)}
    worst = max(max(v for k, v in c.items() if k.endswith("probability_error")) for c in checks.values())
    if worst > config["rehearsal"]["max_probability_error"]:
        raise ValueError(f"Rehearsal does not reproduce saved predictions: {worst}")
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--rehearsal", type=int, metavar="PER_GENRE")
    mode.add_argument("--lock", action="store_true")
    mode.add_argument("--final", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config, labels, pointers = settings()
    if args.lock:
        latest = json.loads((ROOT / "reports/latest_v2_rehearsal.json").read_text())
        rehearsal = f"{latest['folder']}/report.json"
        if sha256_file(ROOT / rehearsal) != latest["report_sha256"]:
            raise ValueError("Rehearsal report changed")
        if (ROOT / FINAL).exists():
            raise ValueError("The final evaluation already started")
        files = lock_files(config, pointers, rehearsal)
        write_json(ROOT / LOCK, {
            "schema_version": 1, "lock": "3", "created_at_utc": datetime.now(UTC).isoformat(), "status": "locked",
            "scope": "systems, weights, code, configs and environment before the single V2 test inference",
            "decisions_before_lock3": config["decisions_before_lock3"], "rehearsal": rehearsal,
            "files_sha256": {p: sha256_file(ROOT / p) for p in files}, "environment": environment(),
            "test_evaluated": False, "v2_test_audio_loaded": False})
        print(f"Lock 3 written: {len(files)} files, sha256 {sha256_file(ROOT / LOCK)}")
        return
    calibration = json.loads((ROOT / config["calibration"]["report"]).read_text(encoding="utf-8"))
    if sha256_file(ROOT / config["calibration"]["report"]) != config["calibration"]["report_sha256"]:
        raise ValueError("Calibration report changed")
    temperature = calibration["final"]["temperature"]
    threshold = calibration["final"]["threshold_top_probability"]
    if args.final:
        lock_sha = verify_lock()
        folder, rows = ROOT / FINAL, final_rows()
        if (folder / "report.json").exists():
            raise ValueError("The single V2 test inference already ran")
        (folder / "tracks").mkdir(parents=True, exist_ok=True)
        marker = folder / "lock3_sha256.txt"
        if marker.exists() and marker.read_text() != lock_sha:
            raise ValueError("Partial evaluation made under another lock")
        marker.write_text(lock_sha)
        with (ROOT / config["decoder_sensitivity"]["csv"]).open(encoding="utf-8", newline="") as stream:
            excluded = {r["track_id"] for r in csv.DictReader(stream) if r["in_final_cohorts"] == "True"}
    else:
        lock_sha, rows, excluded = None, rehearsal_rows(labels, args.rehearsal), set()
        folder = ROOT / "runs/v2-rehearsal" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        (folder / "tracks").mkdir(parents=True)
    tasks = [(r, str(folder)) for r in rows]
    with ProcessPoolExecutor(args.workers, initializer=initialize, initargs=(not args.final,)) as pool:
        saved = dict(pool.map(predict_track, tasks))
    probs = {name: np.array([saved[r["track_id"]][f"p__{name}"].mean(axis=0) for r in rows])
             for name in [*config["v1_systems"], *config["v2_systems"]]}
    with (folder / "predictions.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["track_id", "role", "genre", "artist_id", "system", *(f"p_{label}" for label in labels)])
        writer.writerows([r["track_id"], r["role"], r["genre"], r["artist_id"], s, *(repr(float(v)) for v in p[i])]
                         for s, p in probs.items() for i, r in enumerate(rows))
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(),
        "mode": "final" if args.final else "rehearsal", "lock3_sha256": lock_sha,
        "config_sha256": sha256_file(ROOT / CONFIG), "tracks": len(rows),
        "predictions_sha256": sha256_file(folder / "predictions.csv"),
        "rehearsal_checks": None if args.final else rehearsal_checks(rows, saved, config, pointers, labels),
        **statistics(rows, probs, config, labels, temperature, threshold, excluded),
        "v2_test_audio_loaded": bool(args.final),
    }
    write_json(folder / "report.json", report)
    if not args.final:
        write_json(ROOT / "reports/latest_v2_rehearsal.json", {"folder": folder.relative_to(ROOT).as_posix(),
                                                               "report_sha256": sha256_file(folder / "report.json")})
    print(json.dumps({"mode": report["mode"], "tracks": len(rows), "checks": report["rehearsal_checks"],
                      "hypotheses": {n: {k: h[k] for k in ("point", "low", "high", "p_value")}
                                     for n, h in report["hypotheses"].items()}}, indent=1))


if __name__ == "__main__":
    main()
