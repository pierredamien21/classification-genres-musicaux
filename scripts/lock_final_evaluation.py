"""T6: verify historical evidence, time train inference, publish a distinct lock.

No test audio is decoded. Raw file hashing includes test to detect dataset drift.
Run from the project virtual environment; progress and failure are persisted.
"""

import csv
import gc
import json
import platform
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from music_genre.data.audit import sha256_file, write_json
from music_genre.evaluation.final_protocol import (
    inside,
    inspect_partitions,
    load_locked_manifest,
    validate_evaluation_request,
)

ROOT = Path(__file__).resolve().parents[1]
PROGRESS = ROOT / "reports/FINAL_LOCK_PROGRESS.json"
POINTER = ROOT / "reports/latest_final_evaluation_lock.json"
PROOFS = {
    "T1": "candidate_inference_check", "T2": "robustness_panel",
    "T2a": "noise_sensitivity_diagnostic", "T2b": "block6_robustness_comparison",
    "T3": "bootstrap_validation", "T4": "comparator_inference_check",
    "T5": "cnn_inference_check",
}
RECOUNTS = {
    "T2b": ("BLOCK6_ROBUSTNESS_CHECK", "check_block6_robustness_comparison"),
    "T3": ("BOOTSTRAP_VALIDATION_CHECK", "verify_bootstrap_validation"),
    "T4": ("COMPARATOR_INFERENCE_RECOUNT", "verify_comparator_inference"),
    "T5": ("CNN_INFERENCE_RECOUNT", "verify_cnn_inference"),
}


def read(relative):
    return json.loads(inside(ROOT, relative).read_text("utf-8"))


def progress(phase, **values):
    write_json(PROGRESS, {"status": "running", "phase": phase,
                          "updated_at": datetime.now(UTC).isoformat(), **values})
    print(phase, values, flush=True)


def collect_artifacts(prepared):
    artifacts, proofs = {}, {}

    def pin(name, digest=None):
        digest = digest or sha256_file(inside(ROOT, name))
        if name in artifacts and artifacts[name] != digest:
            raise ValueError(f"Conflicting historical hashes: {name}")
        artifacts[name] = digest

    pin("reports/PREPARATION_EVALUATION_FINALE.json")
    pin(prepared["protocol_path"], prepared["protocol_sha256"])
    pin(prepared["sample"]["path"], prepared["sample"]["sha256"])
    for stage, pointer in PROOFS.items():
        ref = read(f"reports/latest_{pointer}.json")
        name = ref["report"]
        pin(name, ref["report_sha256"])
        proof = read(name)
        if proof.get("status") != "complete" or proof.get("test_evaluated") is not False:
            raise ValueError(f"Invalid prerequisite: {stage}")
        proofs[stage] = name
        for key in ("source_sha256", "source_and_input_sha256"):
            for path, digest in proof.get(key, {}).items():
                pin(path, digest)
        for key in ("archive", "feature_archive"):
            if key in proof:
                pin(proof[key]["path"], proof[key]["sha256"])
        if "preparation_sha256" in proof:
            pin("reports/PREPARATION_EVALUATION_FINALE.json", proof["preparation_sha256"])
    recounts = {}
    for stage, (stem, script) in RECOUNTS.items():
        name = f"reports/{stem}.json"
        report = read(name)
        if report["status"] != "passed" or report["test_evaluated"] is not False:
            raise ValueError(f"Independent recount failed: {stage}")
        ref = report.get("report", report.get("comparison"))
        if ref["report"] != proofs[stage]:
            raise ValueError(f"Independent recount refers to another proof: {stage}")
        pin(ref["report"], ref["report_sha256"])
        pin(f"scripts/{script}.py", report["verification_source_sha256"])
        pin(name)
        recounts[stage] = name
    for model in prepared["runs"].values():
        pin(model["model_path"], model["model_sha256"])
        pin(model["run"]["folder"] + "/report.json", model["run"]["report_sha256"])
        pin(model["run"]["folder"] + "/validation_predictions.csv",
            model["validation_predictions_sha256"])
        pin(model["feature_cache"] + "/report.json", model["feature_report_sha256"])
    for name, digest in prepared["checkpoint"]["sha256"].items():
        pin("artifacts/pretrained/vit_tiny/" + name, digest)
    pin("artifacts/pretrained/vit_tiny/provenance.json")
    baseline = prepared["baseline"]["folder"]
    pin(baseline + "/manifest.csv", prepared["baseline"]["manifest_sha256"])
    pin(baseline + "/report.json", prepared["baseline"]["report_sha256"])
    with inside(ROOT, baseline + "/manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        pin("data/raw/" + row["audio_path"], row["source_sha256"])
    for path in sorted((ROOT / "src/music_genre").rglob("*.py")):
        pin(path.relative_to(ROOT).as_posix())
    pin("scripts/lock_final_evaluation.py")
    pin("tests/test_final_protocol.py")
    pin("pyproject.toml")
    return artifacts, proofs, recounts, rows


def benchmark_train(rows, protocol):
    # Lazy imports: no inference code runs before all input hashes are checked.
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits

    from music_genre.inference.adapters import OriginalBlock6Predictor
    from music_genre.inference.cnn_adapter import CnnPredictor
    from music_genre.inference.linear_adapters import LinearComparatorPredictor
    from music_genre.inference.predict import CandidatePredictor

    selected = {}
    for row in sorted(rows, key=lambda r: r["track_id"]):
        if row["split"] == "train":
            selected.setdefault(row["genre"], row)
    sample = sorted(selected.values(), key=lambda r: r["track_id"])
    if len(sample) != 8:
        raise ValueError("Expected eight train benchmark classes")
    result = {"split": "train", "selection": "lowest_track_id_per_genre",
              "track_ids": [r["track_id"] for r in sample], "systems": {},
              "numerical_threads": {"blas": 1, "torch_linear": 1, "torch_cnn": 4}}
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for name in protocol["linear_systems"] + protocol["cnn_runs"]:
            progress("benchmark_train", system=name, completed_systems=len(result["systems"]))
            start = perf_counter()
            if name == "vit_block6_hpss":
                predictor = CandidatePredictor(ROOT)
            elif name == "vit_block6":
                predictor = OriginalBlock6Predictor(ROOT)
            elif name.startswith("cnn20_seed"):
                predictor = CnnPredictor(ROOT, int(name.removeprefix("cnn20_seed")))
            else:
                predictor = LinearComparatorPredictor(ROOT, name)
            load_seconds = perf_counter() - start
            timings = []
            with threadpool_limits(limits=1):
                for row in sample:
                    start = perf_counter()
                    prediction = predictor.predict_file(inside(ROOT, "data/raw/" + row["audio_path"]))
                    timings.append(perf_counter() - start)
                    scores = np.asarray(prediction["scores"])
                    if not np.isfinite(scores).all() or abs(scores.sum() - 1) > 1e-6:
                        raise ValueError("Invalid benchmark prediction")
            result["systems"][name] = {"load_seconds": load_seconds,
                                       "track_seconds": timings,
                                       "mean_track_seconds": float(np.mean(timings))}
            del predictor
            gc.collect()
    finally:
        torch.set_num_threads(previous_threads)
    estimate = sum(v["mean_track_seconds"] * 778 + v["load_seconds"]
                   for v in result["systems"].values())
    result["estimated_test_inference_seconds"] = estimate
    result["planning_range_seconds"] = [estimate, estimate * 2]
    result["limitations"] = ["Eight train tracks only; first-call overhead included",
                            "Sequential nine exports; no feature sharing",
                            "Excludes bootstrap, hashing, reporting and validation dry run",
                            "Range is a planning margin, not a statistical interval"]
    return result


def main():
    if POINTER.exists():
        raise FileExistsError("A final lock already exists; do not silently replace it")
    started = perf_counter()
    progress("collect_proofs")
    prepared = read("reports/PREPARATION_EVALUATION_FINALE.json")
    protocol = read(prepared["protocol_path"])
    if prepared["test_evaluated"] or protocol["test_evaluation_allowed"]:
        raise ValueError("Expected unopened preparation")
    artifacts, proofs, recounts, rows = collect_artifacts(prepared)
    ids = inspect_partitions(rows, prepared["labels"])
    counts = {s: len(v) for s, v in ids.items()}
    if counts != {"train": 6299, "validation": 780, "test": 778}:
        raise ValueError("Unexpected final cohort")
    for i, (name, digest) in enumerate(artifacts.items()):
        if i % 500 == 0:
            progress("verify_artifacts", checked=i, total=len(artifacts))
        if sha256_file(inside(ROOT, name)) != digest:
            raise ValueError(f"Historical artifact changed: {name}")
    benchmark = benchmark_train(rows, protocol)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    relative = f"reports/final-evaluation-lock-{timestamp}.json"
    locked = {
        "schema_version": 1, "status": "locked", "created_at": timestamp,
        "test_evaluation_allowed": True, "test_evaluated": False, "test_audio_decoded": False,
        "protocol": protocol, "labels": prepared["labels"], "models": prepared["runs"],
        "artifacts": dict(sorted(artifacts.items())),
        "source_files": sorted(p for p in artifacts if p.endswith(".py")),
        "proofs": proofs, "independent_recounts": recounts,
        "dataset": {"manifest": prepared["baseline"]["folder"] + "/manifest.csv",
                    "counts": counts, "track_ids": ids},
        "versions": {"python": platform.python_version(), **{p: version(p) for p in (
            "numpy", "scipy", "librosa", "soundfile", "torch", "timm", "safetensors",
            "scikit-learn", "threadpoolctl", "soxr", "numba", "pandas", "Pillow")}},
        "robustness_disposition": "accepted_limitation_after_T2a_T2b",
        "benchmark": benchmark,
        "execution_gate": "T7 must record its runner hash and successful validation dry run before test",
        "limitations": ["Sensitive to additive noise; no universal robustness winner",
                        "Eight closed-set FMA genres; no gospel or unknown-genre detector",
                        "Validation used for model selection; final test not yet evaluated",
                        "Human listening review not performed; scores not calibrated"],
    }
    # Original preparatory protocol is embedded verbatim as historical provenance.
    # Only this new manifest carries the locked authorization; it never runs T7.
    with inside(ROOT, relative).open("x", encoding="utf-8") as stream:
        json.dump(locked, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    digest = sha256_file(inside(ROOT, relative))
    progress("roundtrip_integrity_check", manifest=relative)
    checked = load_locked_manifest(ROOT, relative, digest)
    validate_evaluation_request(checked, ids["validation"], "validation")
    pointer = {"report": relative, "report_sha256": digest,
               "status": "verified", "artifacts_verified": len(artifacts),
               "elapsed_seconds": perf_counter() - started,
               "validation_cohort_gate": "passed", "test_evaluated": False}
    with POINTER.open("x", encoding="utf-8") as stream:
        json.dump(pointer, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    write_json(PROGRESS, {**pointer, "status": "complete"})
    print(json.dumps(pointer), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        write_json(PROGRESS, {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                              "test_evaluated": False})
        raise
