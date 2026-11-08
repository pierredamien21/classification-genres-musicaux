"""Run extraction, training and comparison sequentially; preserve failure status."""

import argparse
import json
import msvcrt
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from compare_vit_baseline import read_run

from music_genre.data.audit import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()
    status_path = ROOT / "reports/VIT_BLOCK6_HPSS_PIPELINE.json"
    state = {"status": "running", "pid": os.getpid(), "started_at_utc": datetime.now(UTC).isoformat(),
             "stage": "extraction", "test_evaluated": False,
             "fixed_sha256": {p: sha256_file(ROOT / p) for p in (
                 "configs/baseline_vit_block6_hpss.json", "scripts/train_descriptor_baseline.py",
                 "scripts/compare_vit_block6_hpss.py", "scripts/compare_vit_baseline.py",
                 "src/music_genre/models/linear.py", "src/music_genre/evaluation/metrics.py")}}
    with (ROOT / "data/interim/block6_hpss_pipeline.lock").open("a+b") as lock:
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            write_json(status_path, state)
            subprocess.run([sys.executable, "scripts/extract_vit_block6_hpss.py", "--workers", str(args.workers)], cwd=ROOT, check=True)
            if any(sha256_file(ROOT / p) != expected for p, expected in state["fixed_sha256"].items()):
                raise ValueError("Training/comparison configuration or source changed while extracting")
            state["stage"] = "training"
            write_json(status_path, state)
            trained = False
            if (ROOT / "reports/latest_vit_block6_hpss_run.json").exists():
                _, run, _, _, _ = read_run("vit_block6_hpss")
                cache = json.loads((ROOT / "data/features/latest_vit_block6_hpss.json").read_text(encoding="utf-8"))
                config = json.loads((ROOT / "configs/baseline_vit_block6_hpss.json").read_text(encoding="utf-8"))
                trained = (run["feature_cache"] == cache["folder"] and run["feature_report_sha256"] == cache["report_sha256"]
                           and run["config"] == config)
            if not trained:
                subprocess.run([sys.executable, "scripts/train_descriptor_baseline.py", "--features", "vit_block6_hpss"], cwd=ROOT, check=True)
            state["stage"] = "comparison"
            write_json(status_path, state)
            subprocess.run([sys.executable, "scripts/compare_vit_block6_hpss.py"], cwd=ROOT, check=True)
            state.update(status="complete", finished_at_utc=datetime.now(UTC).isoformat())
            write_json(status_path, state)
        except BaseException as error:
            state.update(status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
            write_json(status_path, state)
            raise


if __name__ == "__main__":
    main()
