"""Audit all FMA-small audio; native decoder diagnostics are isolated per worker."""

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from music_genre.data.audit import (
    audit_audio,
    classify_row,
    duplicate_groups,
    parse_decoder_log,
    sha256_file,
    write_json,
)

ROOT = Path(__file__).resolve().parents[1]


def metadata_rows():
    labels = json.loads((ROOT / "configs/labels.json").read_text(encoding="utf-8"))["labels"]
    with (ROOT / "data/raw/fma_metadata/tracks.csv").open(encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        parents, names = next(reader), next(reader)
        next(reader)
        columns = dict(zip(zip(parents, names), range(len(parents))))
        rows = []
        for raw in reader:
            def get(parent, name, record=raw):
                return record[columns[parent, name]]
            if get("set", "subset") != "small":
                continue
            track_id = f"{int(raw[0]):06d}"
            genre = get("track", "genre_top")
            split = {"training": "train", "validation": "validation", "test": "test"}[
                get("set", "split")
            ]
            if genre not in labels:
                raise ValueError(f"Unknown genre: {genre}")
            rows.append({
                "track_id": track_id, "artist_id": get("artist", "id"),
                "album_id": get("album", "id"), "genre": genre,
                "label_index": labels.index(genre), "split": split,
                "audio_path": f"fma_small/{track_id[:3]}/{track_id}.mp3",
                "license": get("track", "license"),
                "metadata_duration_seconds": get("track", "duration"),
            })
    if len(rows) != 8000 or len({r["track_id"] for r in rows}) != 8000:
        raise ValueError("Expected 8000 distinct FMA-small records")
    return sorted(rows, key=lambda r: r["track_id"])


def worker(folder: Path, index: int):
    settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    rows = json.loads((folder / f"input-{index}.json").read_text(encoding="utf-8"))
    raw_root = (ROOT / "data/raw").resolve()
    with (folder / f"results-{index}.jsonl").open("w", encoding="utf-8") as output:
        for row in rows:
            path = (raw_root / row["audio_path"]).resolve()
            if not path.is_relative_to(raw_root):
                raise ValueError("Audio path outside raw data")
            os.write(2, f"AUDIT_BEGIN {row['track_id']}\n".encode())
            row.update(audit_audio(path, settings))
            os.write(2, f"AUDIT_END {row['track_id']}\n".encode())
            output.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            output.flush()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker", type=int)
    parser.add_argument("--folder", type=Path)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args.folder, args.worker)
        return
    if not 1 <= args.workers <= 8:
        parser.error("--workers must be between 1 and 8")
    started = time.perf_counter()
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    folder = ROOT / "data/manifests" / f"audit-{stamp}"
    folder.mkdir(parents=True, exist_ok=False)
    settings = json.loads((ROOT / "configs/audit.json").read_text(encoding="utf-8"))
    write_json(folder / "settings.json", settings)
    rows = metadata_rows()
    processes, logs = [], []
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    print(f"Audit folder: {folder}", flush=True)
    try:
        for index in range(args.workers):
            write_json(folder / f"input-{index}.json", rows[index::args.workers])
            log = (folder / f"decoder-{index}.log").open("wb")
            logs.append(log)
            processes.append(subprocess.Popen(
                [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", str(index),
                 "--folder", str(folder)], stderr=log, env=env,
            ))
        while any(p.poll() is None for p in processes):
            finished = sum(
                p.read_bytes().count(b"\n")
                for p in folder.glob("results-*.jsonl")
            )
            print(f"Audit: approximately {finished}/8000 completed", flush=True)
            time.sleep(10)
        if any(p.returncode for p in processes):
            raise RuntimeError(f"Audit worker failed; inspect logs in {folder}")
    finally:
        for process in processes:
            if process.poll() is None:
                process.terminate()
                process.wait()
        for log in logs:
            log.close()
    audited = []
    for index in range(args.workers):
        diagnostics = parse_decoder_log(
            (folder / f"decoder-{index}.log").read_text(encoding="utf-8", errors="replace")
        )
        for line in (folder / f"results-{index}.jsonl").read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            row["decoder_diagnostic"] = diagnostics.get(row["track_id"], "")
            if row["decoder_diagnostic"]:
                row["flags"].append("decoder_diagnostic")
            if not row["artist_id"]:
                row["flags"].append("missing_artist_id")
            if not row["license"]:
                row["flags"].append("missing_license")
            audited.append(row)
    audited.sort(key=lambda r: r["track_id"])
    if [r["track_id"] for r in audited] != [r["track_id"] for r in rows]:
        raise RuntimeError("Incomplete audit or repeated tracks")
    duplicates = {field: duplicate_groups(audited, field)
                  for field in ("source_sha256", "pcm_sha256")}
    by_id = {r["track_id"]: r for r in audited}
    for field, groups in duplicates.items():
        for group in groups:
            flag = f"duplicate_{field}" + ("_cross_split" if group["cross_split"] else "")
            for track_id in group["track_ids"]:
                by_id[track_id]["flags"].append(flag)
    for row in audited:
        classify_row(row)
    artists = {split: {r["artist_id"] for r in audited
                       if r["split"] == split and r["artist_id"]}
               for split in ("train", "validation", "test")}
    overlaps = {f"{a}__{b}": sorted(artists[a] & artists[b])
                for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))}
    manifest = folder / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(audited[0]))
        writer.writeheader()
        for row in audited:
            writer.writerow({**row, "flags": "|".join(row["flags"])})
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(),
        "audit_complete": True, "tracks": len(audited),
        "elapsed_seconds": time.perf_counter() - started, "workers": args.workers,
        "settings": settings, "manifest_sha256": sha256_file(manifest),
        "metadata_sha256": sha256_file(ROOT / "data/raw/fma_metadata/tracks.csv"),
        "labels_sha256": sha256_file(ROOT / "configs/labels.json"),
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in
                          [Path(__file__).resolve(), ROOT / "src/music_genre/data/audit.py"]},
        "versions": {"python": sys.version, "numpy": np.__version__,
                     "soundfile": sf.__version__, "libsndfile": sf.__libsndfile_version__},
        "status_counts": dict(Counter(r["status"] for r in audited)),
        "decode_counts": dict(Counter(r["decode_status"] for r in audited)),
        "flag_counts": dict(Counter(flag for r in audited for flag in r["flags"])),
        "split_status_counts": {split: dict(Counter(r["status"] for r in audited if r["split"] == split))
                                for split in artists},
        "genre_status_counts": {genre: dict(Counter(r["status"] for r in audited if r["genre"] == genre))
                                for genre in sorted({r["genre"] for r in audited})},
        "license_counts": dict(Counter(r["license"] for r in audited)),
        "artist_overlap": overlaps, "duplicates": duplicates,
        "review_track_ids": [r["track_id"] for r in audited if r["status"] == "needs_review"],
        "excluded": [{"track_id": r["track_id"], "reason": r["exclusion_reason"], "error": r["error"]}
                     for r in audited if r["status"] == "excluded"],
        "limitations": [
            "Exact source and native float32 PCM hashes only; no perceptual near-duplicate search.",
            "Silence measured on one-second channel-preserving blocks; thresholds are heuristics.",
            "Near-full-scale samples do not prove clipping; flagged tracks require review.",
            "Needs-review rows are not approved for training; no split changed, no source removed.",
            "Licenses copied from metadata, not individually legally reviewed.",
            "Global technical audit only; no test-set model evaluation or parameter selection.",
        ],
    }
    write_json(folder / "report.json", report)
    write_json(ROOT / "data/manifests/latest_audit.json",
               {"folder": folder.relative_to(ROOT).as_posix(), "manifest_sha256": report["manifest_sha256"]})
    print(json.dumps({k: report[k] for k in
                      ("tracks", "decode_counts", "status_counts", "flag_counts", "elapsed_seconds")},
                     indent=2), flush=True)
    print(f"Report: {folder / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
