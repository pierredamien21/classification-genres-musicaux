"""Compute local Chromaprint signatures and retrieve conservative duplicate candidates."""

import csv
import json
import math
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

from music_genre.data.audit import sha256_file, write_json
from music_genre.data.fingerprints import find_candidates, informative

ROOT = Path(__file__).resolve().parents[1]


def main():
    policy_path = ROOT / "configs/review_policy.json"
    policy = json.loads(policy_path.read_text(encoding="utf-8"))
    settings = policy["fingerprint"]
    executable = ROOT / settings["executable"]
    if sha256_file(executable) != settings["executable_sha256"]:
        raise ValueError("Unverified fpcalc executable")
    pointer = json.loads((ROOT / "data/manifests/latest_audit.json").read_text(encoding="utf-8"))
    audit = (ROOT / pointer["folder"]).resolve()
    if not audit.is_relative_to((ROOT / "data/manifests").resolve()):
        raise ValueError("Audit outside manifests")
    manifest = audit / "manifest.csv"
    if sha256_file(manifest) != pointer["manifest_sha256"]:
        raise ValueError("Audit manifest changed")
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    folder = ROOT / "data/manifests" / ("fingerprints-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    folder.mkdir(exist_ok=False)
    write_json(folder / "policy.json", policy)
    raw_root = (ROOT / "data/raw").resolve()
    started = time.perf_counter()

    def calculate(row):
        path = (raw_root / row["audio_path"]).resolve()
        if not path.is_relative_to(raw_root) or sha256_file(path) != row["source_sha256"]:
            raise ValueError(f"Source missing or changed: {row['track_id']}")
        result = {"track_id": row["track_id"], "fingerprint": [], "duration": None,
                  "returncode": -1, "diagnostic": ""}
        try:
            process = subprocess.run(
                [str(executable), "-raw", "-json", "-algorithm", str(settings["algorithm"]),
                 "-length", str(settings["max_seconds"]), str(path)],
                check=False, capture_output=True, timeout=settings["timeout_seconds"],
            )
            result.update(returncode=process.returncode,
                          diagnostic=process.stderr.decode("utf-8", errors="replace").strip())
            if process.stdout.strip():
                decoded = json.loads(process.stdout)
                words = decoded["fingerprint"]
                duration = float(decoded["duration"])
                if (not math.isfinite(duration) or duration <= 0
                        or not all(isinstance(v, int) and 0 <= v < 2**32 for v in words)):
                    raise ValueError("Malformed fingerprint")
                result.update(fingerprint=words, duration=duration)
        except (subprocess.TimeoutExpired, ValueError, KeyError, OSError) as error:
            result.update(returncode=-1, diagnostic=f"{result['diagnostic']}\n{error}".strip())
        return result

    records = []
    print(f"Fingerprint folder: {folder}", flush=True)
    with ((folder / "records.jsonl").open("w", encoding="utf-8") as output,
          ThreadPoolExecutor(max_workers=4) as pool):
        for index, record in enumerate(pool.map(calculate, rows), start=1):
            records.append(record)
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            if index % 250 == 0:
                print(f"Fingerprint extraction: {index}/{len(rows)}", flush=True)
    print("Retrieving and comparing fingerprint candidates...", flush=True)
    candidates = find_candidates(records, settings)
    by_id = {r["track_id"]: r for r in rows}
    for candidate in candidates:
        a, b = by_id[candidate["a"]], by_id[candidate["b"]]
        candidate.update(
            exact_pcm=bool(a["pcm_sha256"] and a["pcm_sha256"] == b["pcm_sha256"]),
            cross_split=a["split"] != b["split"], same_genre=a["genre"] == b["genre"],
        )
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(),
        "audit_folder": pointer["folder"], "audit_sha256": pointer["manifest_sha256"],
        "policy_sha256": sha256_file(policy_path), "policy": policy,
        "records_sha256": sha256_file(folder / "records.jsonl"),
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in
                          [Path(__file__).resolve(), ROOT / "src/music_genre/data/fingerprints.py"]},
        "tool_version": subprocess.check_output([str(executable), "-version"]).decode().strip(),
        "tracks": len(records), "elapsed_seconds": time.perf_counter() - started,
        "nonzero_exit_ids": [r["track_id"] for r in records if r["returncode"] != 0],
        "diagnostic_ids": [r["track_id"] for r in records if r["diagnostic"]],
        "low_information_ids": [r["track_id"] for r in records
                                if not informative(r["fingerprint"], settings)],
        "candidates": candidates,
        "limitations": [
            "Local heuristic candidate retrieval, not the AcoustID matching service.",
            "Requires shared exact fingerprint words; common words discarded.",
            "Aligned comparison requires >=120 fingerprint words of overlap.",
            "Short overlaps, strong edits, shifted non-overlapping excerpts can be missed.",
            "A fingerprint candidate is not proof of duplicate content.",
            "No audio or fingerprint uploaded; no model score or genre used for retrieval.",
        ],
    }
    write_json(folder / "report.json", report)
    write_json(ROOT / "data/manifests/latest_fingerprints.json",
               {"folder": folder.relative_to(ROOT).as_posix(),
                "report_sha256": sha256_file(folder / "report.json")})
    print(json.dumps({k: report[k] for k in
                      ("tracks", "elapsed_seconds", "nonzero_exit_ids", "low_information_ids")},
                     indent=2), flush=True)
    print(f"Candidates: {len(candidates)}; report: {folder / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
