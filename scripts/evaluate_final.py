"""Run/resume frozen inference; validation must pass before any test decoding.

Examples (from project root):
  python scripts/evaluate_final.py --split validation --output data/interim/final-validation-v1
  python scripts/evaluate_final.py --split test --output data/interim/final-test-v1 \
      --validation data/interim/final-validation-v1
No training or parameter selection; reporting/bootstrap is a separate stage.
"""

import argparse
import csv
import gc
import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

from music_genre.data.audit import sha256_file
from music_genre.evaluation.final_evaluation import (
    PredictionStore,
    atomic_json,
    check_validation_evidence,
    compare_validation,
    exclusive_campaign,
)
from music_genre.evaluation.final_protocol import (
    inside,
    load_locked_manifest,
    validate_evaluation_request,
)

ROOT = Path(__file__).resolve().parents[1]
RUNNER_SOURCES = ["scripts/evaluate_final.py", "src/music_genre/evaluation/final_evaluation.py"]


def read_json(path):
    return json.loads(path.read_text("utf-8"))


def read_csv(path):
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def build_campaign(locked, lock_digest, rows, split, sources):
    ids = [r["track_id"] for r in rows if r["split"] == split]
    validate_evaluation_request(locked, ids, split)
    model_ids = {}
    for stage in ("T4", "T5"):
        for name, result in read_json(inside(ROOT, locked["proofs"][stage]))["systems"].items():
            model_ids[name] = result["model_id"]
    names = locked["protocol"]["linear_systems"] + locked["protocol"]["cnn_runs"]
    if set(model_ids) != set(names):
        raise ValueError("Verified model roster differs")
    return {"schema_version": 1, "split": split, "lock_sha256": lock_digest,
            "runner_sources": sources, "labels": locked["labels"],
            "models": {name: model_ids[name] for name in names}, "track_ids": sorted(ids),
            "source_sha256": {r["track_id"]: r["source_sha256"] for r in rows if r["split"] == split},
            "settings": {"mode": "fma", "aggregation": "mean_of_window_probabilities",
                         "blas_threads": 1, "torch_linear_threads": 1, "torch_cnn_threads": 4,
                         "validation_probability_atol": 1e-6, "validation_same_argmax": True}}


def predictor_for(name):
    from music_genre.inference.adapters import OriginalBlock6Predictor
    from music_genre.inference.cnn_adapter import CnnPredictor
    from music_genre.inference.linear_adapters import LinearComparatorPredictor
    from music_genre.inference.predict import CandidatePredictor

    if name == "vit_block6_hpss":
        return CandidatePredictor(ROOT)
    if name == "vit_block6":
        return OriginalBlock6Predictor(ROOT)
    if name.startswith("cnn20_seed"):
        return CnnPredictor(ROOT, int(name.removeprefix("cnn20_seed")))
    return LinearComparatorPredictor(ROOT, name)


def verify_runner(sources):
    for name, digest in sources.items():
        if sha256_file(inside(ROOT, name)) != digest:
            raise ValueError(f"Runner source changed during execution: {name}")


def run(args, folder):
    started = perf_counter()

    def progress(phase, **values):
        entry = {"status": "running", "phase": phase, "split": args.split,
                 "pid": os.getpid(), "updated_at": datetime.now(UTC).isoformat(),
                 "invocation_elapsed_seconds": perf_counter() - started, **values}
        atomic_json(folder / "progress.json", entry)
        with (folder / "events.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry) + "\n")
        print(json.dumps(entry), flush=True)

    progress("integrity_preflight")
    sources = {name: sha256_file(inside(ROOT, name)) for name in RUNNER_SOURCES}
    ref = read_json(ROOT / "reports/latest_final_evaluation_lock.json")
    locked = load_locked_manifest(ROOT, ref["report"], ref["report_sha256"])
    rows = read_csv(inside(ROOT, locked["dataset"]["manifest"]))
    campaign = build_campaign(locked, ref["report_sha256"], rows, args.split, sources)
    validation_ref = None
    if args.split == "test":
        if args.validation is None:
            raise ValueError("Test requires --validation with a successful complete dry run")
        validation_folder = inside(ROOT, args.validation)
        expected = build_campaign(locked, ref["report_sha256"], rows, "validation", sources)
        check_validation_evidence(validation_folder, expected)
        validation_ref = {"path": args.validation + "/completion.json",
                          "sha256": sha256_file(validation_folder / "completion.json")}
        campaign["validation_evidence"] = validation_ref
    elif args.validation is not None:
        raise ValueError("--validation evidence is only used for test campaigns")
    store = PredictionStore(folder, campaign)
    if (folder / "completion.json").exists():
        completion = read_json(folder / "completion.json")
        if (completion["campaign_sha256"] != store.identity
                or completion["predictions"] != store.receipts()):
            raise ValueError("Completed campaign changed")
        if args.split == "validation":
            check_validation_evidence(folder, campaign)
        progress("already_complete", completed=len(completion["predictions"]))
        atomic_json(folder / "progress.json", {"status": "complete", "split": args.split,
                                               "completion_sha256": sha256_file(folder / "completion.json")})
        return
    invocation = {"at": datetime.now(UTC).isoformat(), "pid": os.getpid(),
                  "argv": sys.argv, "campaign_sha256": store.identity,
                  "executable": sys.executable}
    with (folder / "invocations.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(invocation) + "\n")
    cohort = {r["track_id"]: r for r in rows if r["split"] == args.split}
    # Validate every existing record before using it or decoding more audio.
    completed = {name: sum(store.read(name, track) is not None for track in campaign["track_ids"])
                 for name in campaign["models"]}
    if args.split == "test" and not (folder / "test_access_started.json").exists():
        atomic_json(folder / "test_access_started.json", {
            "at": datetime.now(UTC).isoformat(), "campaign_sha256": store.identity,
            "validation_evidence": validation_ref,
            "meaning": "Test access authorized; marker precedes first decode, even if it fails"})
    import torch
    from threadpoolctl import threadpool_limits

    previous_threads = torch.get_num_threads()
    systems = {}
    total = len(campaign["models"]) * len(cohort)
    try:
        torch.set_num_threads(1)
        for name in campaign["models"]:
            pending = [t for t in campaign["track_ids"] if store.read(name, t) is None]
            progress("system_start", system=name, completed_by_system=completed, total=total)
            predictor = predictor_for(name) if pending else None
            if predictor is not None and predictor.model_id != campaign["models"][name]:
                raise ValueError("Runtime model identity differs from verified export")
            with threadpool_limits(limits=1):
                for i, track in enumerate(pending):
                    source = inside(ROOT, "data/raw/" + cohort[track]["audio_path"])
                    if sha256_file(source) != campaign["source_sha256"][track]:
                        raise ValueError(f"Audio changed before decode: {track}")
                    result = predictor.predict_file(source, mode="fma")
                    store.save(name, track, result)
                    completed[name] += 1
                    if (i + 1) % 10 == 0 or i + 1 == len(pending):
                        remaining = sum((len(cohort) - completed[n]) *
                                        locked["benchmark"]["systems"][n]["mean_track_seconds"]
                                        for n in campaign["models"])
                        progress("inference", system=name, last_track_id=track,
                                 completed=sum(completed.values()), total=total,
                                 completed_by_system=completed,
                                 remaining_inference_seconds_from_train_benchmark=remaining)
            del predictor
            gc.collect()
            if args.split == "validation":
                reference = inside(ROOT, locked["models"][name]["run"]["folder"] +
                                   "/validation_predictions.csv")
                if sha256_file(reference) != locked["artifacts"][reference.relative_to(ROOT).as_posix()]:
                    raise ValueError("Validation reference changed")
                systems[name] = compare_validation(store, name, read_csv(reference))
                atomic_json(folder / "validation_checks.json", systems)
            else:
                systems[name] = {"tracks": completed[name]}
        progress("final_integrity_check", completed=total, total=total)
        verify_runner(sources)
        # Repeat all scientific pins after the run: do not seal a moving environment.
        load_locked_manifest(ROOT, ref["report"], ref["report_sha256"])
        receipts = store.receipts()
        completion = {"status": "validation_passed" if args.split == "validation" else "inference_complete",
                      "campaign_sha256": store.identity, "lock": ref, "systems": systems,
                      "predictions": receipts, "validation_evidence": validation_ref,
                      "completed_at": datetime.now(UTC).isoformat(),
                      "last_invocation_seconds": perf_counter() - started,
                      "test_evaluated": args.split == "test", "statistics_reported": False}
        atomic_json(folder / "completion.json", completion)
        if args.split == "validation":
            check_validation_evidence(folder, campaign)
        atomic_json(folder / "progress.json", {
            "status": "complete", "split": args.split, "completed": total, "total": total,
            "updated_at": datetime.now(UTC).isoformat(),
            "completion_sha256": sha256_file(folder / "completion.json"),
            "test_evaluated": args.split == "test"})
        print(json.dumps({"status": "complete", "folder": str(folder)}), flush=True)
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("validation", "test"), required=True)
    parser.add_argument("--output", required=True, help="Project-relative new/resumable data/interim folder")
    parser.add_argument("--validation", help="Project-relative completed validation campaign")
    args = parser.parse_args()
    folder = inside(ROOT, args.output)
    interim = ROOT / "data/interim"
    if folder == interim or not folder.is_relative_to(interim):
        raise ValueError("Output must be a dedicated folder inside data/interim")
    with exclusive_campaign(folder):
        try:
            run(args, folder)
        except BaseException as exc:
            atomic_json(folder / "progress.json", {
                "status": "failed", "split": args.split, "pid": os.getpid(),
                "updated_at": datetime.now(UTC).isoformat(),
                "error": f"{type(exc).__name__}: {exc}",
                "test_access_started": (folder / "test_access_started.json").exists()})
            with (folder / "errors.log").open("a", encoding="utf-8") as stream:
                traceback.print_exc(file=stream)
            raise


if __name__ == "__main__":
    main()
