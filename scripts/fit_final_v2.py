"""Lock 3 preparation: refit the V2 candidate head on the whole development set.

  python scripts/fit_final_v2.py --run runs/v2-cv/dev_maest-layer12-<timestamp> --threads 8

Uses the run's cache, column family, CV config and final C (user decision of 2026-11-11: one
logistic head on all 7 857 tracks). Exports a LinearProbabilityModel and checks that it gives the
same probabilities as the fitted scikit-learn pipeline. Development features only.
"""

import argparse
import json
import sys
import warnings
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cv_v2 import classifier, load_cache

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.crossval import macro_f1
from music_genre.models.linear import LinearProbabilityModel

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", required=True)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    run = (ROOT / args.run).resolve()
    result = json.loads((run / "report.json").read_text(encoding="utf-8"))
    if result.get("head", "logistic") != "logistic" or result.get("augment_cache"):  # older runs: no "head"
        raise ValueError("Only the clean logistic head is supported")
    labels = json.loads((ROOT / "configs/labels.json").read_text())["labels"]
    pointer, report, index, x = load_cache(result["features"])
    if pointer["folder"] != result["feature_cache"] or pointer["report_sha256"] != result["feature_report_sha256"]:
        raise ValueError("Clean cache differs from the run's")
    columns = [i for i, f in enumerate(report["feature_families"]) if f == result["family"]]
    x = x[:, columns]
    if x.shape != (result["windows"], result["feature_dimensions"]):
        raise ValueError("Feature matrix differs from the run's")
    tracks = np.array([r["track_id"] for r in index])
    y = np.array([int(r["label_index"]) for r in index])
    c = result["final_c"]
    pipeline = Pipeline([("scale", StandardScaler()), ("classifier", classifier(result["config"], c))])
    with threadpool_limits(limits=args.threads), warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        pipeline.fit(x, y)
    if any(issubclass(w.category, ConvergenceWarning) for w in captured):
        raise RuntimeError("Final head did not converge")
    scale, head = pipeline.named_steps["scale"], pipeline.named_steps["classifier"]
    if not np.array_equal(head.classes_, np.arange(len(labels))):
        raise ValueError("Unexpected class order")
    model = LinearProbabilityModel(scale.mean_, scale.scale_, head.coef_, head.intercept_, head.classes_)
    exported = model.predict_proba(x)
    export_error = float(np.max(np.abs(exported - pipeline.predict_proba(x))))
    if export_error > 1e-9:
        raise ValueError(f"Export changed probabilities: {export_error}")
    folder = ROOT / "runs/v2-final" / f"{result['system']}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}"
    folder.mkdir(parents=True)
    model.save(folder / "model.npz")
    restored = LinearProbabilityModel.load(folder / "model.npz").predict_proba(x)
    if not np.array_equal(restored, exported):
        raise ValueError("Saved model differs")
    summary = {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(), "system": result["system"],
        "decision": "one logistic head refitted on all development windows (user choice, 2026-11-11)",
        "cv_run": run.relative_to(ROOT).as_posix(), "cv_report_sha256": sha256_file(run / "report.json"),
        "feature_cache": pointer["folder"], "feature_report_sha256": pointer["report_sha256"],
        "family": result["family"], "columns": len(columns), "c": c, "config": result["config"],
        "windows": len(x), "tracks": len(set(tracks)),
        "training_track_macro_f1": macro_f1(tracks, y, exported),
        "export_max_probability_error": export_error, "model_sha256": sha256_file(folder / "model.npz"),
        "source_sha256": {p: sha256_file(ROOT / p) for p in (
            "scripts/fit_final_v2.py", "scripts/cv_v2.py", "src/music_genre/models/linear.py")},
        "versions": {n: version(n) for n in ("numpy", "scipy", "scikit-learn")},
        "v2_test_audio_loaded": False,
    }
    write_json(folder / "report.json", summary)
    write_json(ROOT / "reports/latest_v2_final_head.json",
               {"folder": folder.relative_to(ROOT).as_posix(), "report_sha256": sha256_file(folder / "report.json")})
    print(json.dumps({k: summary[k] for k in ("system", "c", "training_track_macro_f1", "export_max_probability_error")},
                     indent=1))


if __name__ == "__main__":
    main()
