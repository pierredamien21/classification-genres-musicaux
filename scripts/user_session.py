"""Explicit user launch, detached supervision and durable status. No implicit training."""

import argparse
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import psutil

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from music_genre.evaluation.final_evaluation import atomic_json, exclusive_campaign


def process_matches(identity):
    if not identity:
        return False
    try:
        return psutil.Process(identity["pid"]).create_time() == identity["created"]
    except psutil.NoSuchProcess:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--stop-after-epoch", type=int)
    args = parser.parse_args()
    name = "cnn_user_smoke" if args.smoke else "cnn_user"
    job = ROOT / "reports" / ("USER_SMOKE_JOB.json" if args.smoke else "USER_TRAINING_JOB.json")
    if not (args.launch or args.worker):
        if not job.exists():
            print("Aucun entraînement lancé.")
            return
        state = json.loads(job.read_text(encoding="utf-8"))
        if state["status"] in {"starting", "running"} and not any(
                process_matches(state.get(key)) for key in ("supervisor", "training_process")):
            state["status"] = "interrupted"
            state["next_action"] = "Inspect logs; resume the last complete checkpoint with --resume."
            atomic_json(job, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        return
    if args.launch:
        log = ROOT / "data/interim" / (name + "-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ") + ".log")
        command = [sys.executable, str(Path(__file__).resolve()), "--worker"]
        command += [arg for arg in sys.argv[1:] if arg != "--launch"]
        with log.open("w", encoding="utf-8") as output:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                       stdout=output, stderr=subprocess.STDOUT,
                                       creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        print(f"Superviseur PID {process.pid}. Journal : {log}. État : {job}")
        return
    with exclusive_campaign(ROOT / "data/interim" / (name + "-supervisor")):
        if job.exists():
            previous = json.loads(job.read_text(encoding="utf-8"))
            for key in ("supervisor", "training_process"):
                if process_matches(previous.get(key)):
                    raise RuntimeError("Une session identifiée est encore active. Aucun doublon lancé.")
        pointer = ROOT / "reports" / f"latest_{name}_training.json"
        command = [sys.executable, "-u", str(ROOT / "scripts/train_user_cnn.py")]
        if args.smoke:
            command.append("--smoke")
        folder = None
        if pointer.exists():
            folder = ROOT / json.loads(pointer.read_text(encoding="utf-8"))["folder"]
            if not args.smoke and not args.resume:
                raise RuntimeError("Une session existe déjà. Vérifier ses sorties ou choisir --resume.")
        if args.resume:
            if folder is None:
                raise RuntimeError("Aucune session à reprendre.")
            progress = folder / "progress.json"
            if progress.exists() and json.loads(progress.read_text())["status"] == "complete":
                raise RuntimeError("Session déjà terminée : aucune reprise nécessaire.")
            command += ["--resume", str(folder)]
        if args.stop_after_epoch:
            command += ["--stop-after-epoch", str(args.stop_after_epoch)]
        state = {"status": "starting", "started_at": datetime.now(UTC).isoformat(),
                 "supervisor": {"pid": os.getpid(), "created": psutil.Process().create_time()},
                 "command": command, "smoke": args.smoke, "test_evaluated": False}
        atomic_json(job, state)
        env = {**os.environ, "PYTHONPATH": str(ROOT / "src"), "PYTHONUNBUFFERED": "1"}
        with subprocess.Popen(command, cwd=ROOT, env=env) as training:
            state.update(status="running", training_process={
                "pid": training.pid, "created": psutil.Process(training.pid).create_time()})
            atomic_json(job, state)
            exit_code = training.wait()
        state.update(exit_code=exit_code, finished_at=datetime.now(UTC).isoformat(),
                     status="failed" if exit_code else "finished")
        if pointer.exists():
            state["folder"] = json.loads(pointer.read_text(encoding="utf-8"))["folder"]
            progress = ROOT / state["folder"] / "progress.json"
            if exit_code == 0 and progress.exists():
                state["status"] = json.loads(progress.read_text())["status"]
        atomic_json(job, state)
        print(json.dumps(state, ensure_ascii=False, indent=2))
        if exit_code:
            raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
