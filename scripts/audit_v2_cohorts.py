"""Technical audit of the downloaded V2 tracks, pre-registered replacements, then lock 2.

Uses only the V1 tools and thresholds (configs/audit.json, configs/review_policy.json):
audit_audio, fpcalc 1.6.2 and find_candidates. No model, prediction or listening is used.
A track fails on: decode failure, any non-benign audit flag, missing licence, fingerprint
failure or diagnostic, low-information fingerprint, or an exact/near duplicate of any
FMA-small track or other V2 track. Failed tracks are replaced by the next passing spare
of the same role and genre, in rank order (docs/PROTOCOLE_V2.md). Amendment 1 adds Pop
spares ranked after the original ones (scripts/amend_v2_pop.py).

Writes reports/v2/audit.csv, reports/v2/final_cohorts.csv and reports/v2/lock2.json.
"""

import csv
import json
import subprocess
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from music_genre.data.audit import audit_audio, sha256_file, write_json
from music_genre.data.fingerprints import find_candidates, informative

ROOT = Path(__file__).resolve().parents[1]
BENIGN = {"near_full_scale"}
PER_GENRE = 64
SOURCES = (("reports/v2/cohorts.csv", "data/manifests/v2_audio.json"),
           ("reports/v2/cohorts_amendment.csv", "data/manifests/v2_audio_amendment.json"))


def fingerprint(executable, settings, path):
    process = subprocess.run([str(executable), "-raw", "-json", "-algorithm", str(settings["algorithm"]),
                              "-length", str(settings["max_seconds"]), str(path)],
                             check=False, capture_output=True, timeout=settings["timeout_seconds"])
    diagnostic = process.stderr.decode("utf-8", errors="replace").strip()
    try:
        words = json.loads(process.stdout)["fingerprint"] if process.stdout.strip() else []
    except (ValueError, KeyError) as error:
        return {"returncode": -1, "fingerprint": [], "diagnostic": f"{diagnostic}\n{error}".strip()}
    return {"returncode": process.returncode, "fingerprint": words, "diagnostic": diagnostic}


def replace(rows, passed):
    """Keep passing primaries; fill each failed primary with the next passing spare."""
    final = []
    for key in sorted({(r["role"].removesuffix("_spare"), r["genre"]) for r in rows}):
        group = sorted((r for r in rows if (r["role"].removesuffix("_spare"), r["genre"]) == key),
                       key=lambda r: int(r["rank"]))
        primaries = [r for r in group if not r["role"].endswith("_spare")]
        spares = [r for r in group if r["role"].endswith("_spare") and passed[r["track_id"]]]
        kept = [r for r in primaries if passed[r["track_id"]]]
        kept += spares[:len(primaries) - len(kept)]
        final += [{**r, "role": key[0], "replacement": r["role"].endswith("_spare")} for r in kept]
    return final


def main():
    audit_settings = json.loads((ROOT / "configs/audit.json").read_text())
    policy = json.loads((ROOT / "configs/review_policy.json").read_text())["fingerprint"]
    executable = ROOT / policy["executable"]
    if sha256_file(executable) != policy["executable_sha256"]:
        raise ValueError("Unverified fpcalc executable")
    rows, hashes = [], {}
    for cohorts, manifest in SOURCES:
        with (ROOT / cohorts).open(encoding="utf-8", newline="") as stream:
            rows += list(csv.DictReader(stream))
        download = json.loads((ROOT / manifest).read_text())
        if download["cohorts_sha256"] != sha256_file(ROOT / cohorts):
            raise ValueError(f"Downloaded audio belongs to other cohorts: {manifest}")
        hashes.update({f["zip_member"]: f["sha256"] for f in download["files"]})
    licences = pd.read_csv(ROOT / "data/raw/fma_metadata/tracks.csv", index_col=0, header=[0, 1],
                           low_memory=False)[("track", "license")]

    def check(row):
        path = ROOT / "data/raw/fma_large" / Path(row["zip_member"]).relative_to("fma_large")
        if sha256_file(path) != hashes[row["zip_member"]]:
            raise ValueError(f"Audio changed since download: {row['track_id']}")
        result = audit_audio(path, audit_settings)
        result.update(fingerprint(executable, policy, path))
        return row["track_id"], result

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = dict(pool.map(check, rows))
    pointer = json.loads((ROOT / "data/manifests/latest_fingerprints.json").read_text())
    with (ROOT / pointer["folder"] / "records.jsonl").open(encoding="utf-8") as stream:
        dev_records = [json.loads(line) for line in stream]
    audit_pointer = json.loads((ROOT / "data/manifests/latest_audit.json").read_text())
    with (ROOT / audit_pointer["folder"] / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        dev_pcm = {r["pcm_sha256"] for r in csv.DictReader(stream) if r["pcm_sha256"]}
    v2_ids = {r["track_id"] for r in rows}
    records = dev_records + [{"track_id": t, **{k: results[t][k] for k in ("returncode", "fingerprint", "diagnostic")}}
                             for t in v2_ids]
    duplicates = Counter()
    for pair in find_candidates(records, policy):
        for track in (pair["a"], pair["b"]):
            if track in v2_ids:
                duplicates[track] += 1
    v2_pcm = Counter(results[t]["pcm_sha256"] for t in v2_ids if results[t]["pcm_sha256"])
    audit_rows, passed = [], {}
    for row in rows:
        result = results[row["track_id"]]
        reasons = sorted(set(result["flags"]) - BENIGN)
        if result["decode_status"] != "ok":
            reasons.append("decode_failed")
        if not isinstance(licences.get(int(row["track_id"])), str):
            reasons.append("missing_license")
        if result["returncode"] != 0 or not result["fingerprint"]:
            reasons.append("fingerprint_failed")
        if result["diagnostic"]:
            reasons.append("decoder_diagnostic")
        if not informative(result["fingerprint"], policy):
            reasons.append("low_information_fingerprint")
        if result["pcm_sha256"] and (result["pcm_sha256"] in dev_pcm or v2_pcm[result["pcm_sha256"]] > 1):
            reasons.append("exact_pcm_duplicate")
        if duplicates[row["track_id"]]:
            reasons.append("fingerprint_duplicate_candidate")
        passed[row["track_id"]] = not reasons
        audit_rows.append({"track_id": row["track_id"], "role": row["role"], "genre": row["genre"],
                           "duration_seconds": round(result["duration_seconds"], 3),
                           "sample_rate": result["sample_rate"], "channels": result["channels"],
                           "flags": "|".join(result["flags"]), "reasons": "|".join(reasons),
                           "passed": not reasons})
    final = replace(rows, passed)
    assert len({r["artist_id"] for r in final}) == len(final) and all(passed[r["track_id"]] for r in final)
    out = ROOT / "reports/v2"
    with (out / "audit.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(audit_rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(audit_rows)
    with (out / "final_cohorts.csv").open("w", encoding="utf-8", newline="") as stream:
        fields = ["role", "genre", "track_id", "artist_id", "rank", "replacement", "audio_path", "sha256"]
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows({**{k: r[k] for k in fields[:6]},
                          "audio_path": r["zip_member"],  # relative to data/raw
                          "sha256": hashes[r["zip_member"]]} for r in final)
    counts = Counter((r["role"], r["genre"]) for r in final)
    reasons = Counter(reason for r in audit_rows for reason in filter(None, r["reasons"].split("|")))
    lock = {"schema_version": 1, "lock": 2, "created_at_utc": datetime.now(UTC).isoformat(),
            "status": "locked_before_any_v2_model_run",
            "files_sha256": {p: sha256_file(ROOT / p) for p in (
                "configs/protocol_v2.json", "docs/PROTOCOLE_V2.md", "reports/v2/selection.json",
                "reports/v2/cohorts.csv", "reports/v2/cohorts_amendment.csv", "reports/v2/amendment.json",
                "reports/v2/dev_folds.csv", "reports/v2/power.json", "scripts/amend_v2_pop.py",
                "reports/v2/audit.csv", "reports/v2/final_cohorts.csv", "configs/audit.json",
                "configs/review_policy.json", "scripts/audit_v2_cohorts.py", "scripts/fetch_v2_audio.py")},
            "download_manifests_sha256": {m: sha256_file(ROOT / m) for _, m in SOURCES},
            "fpcalc": subprocess.check_output([str(executable), "-version"]).decode().strip(),
            "audited_tracks": len(rows), "failed_tracks": sum(not p for p in passed.values()),
            "failure_reasons": dict(sorted(reasons.items())),
            "final_counts": {f"{role}/{genre}": n for (role, genre), n in sorted(counts.items())},
            "replacements_used": sum(r["replacement"] for r in final),
            "models_run": False, "predictions_seen": False, "listening": False}
    write_json(out / "lock2.json", lock)
    print(json.dumps({k: lock[k] for k in ("audited_tracks", "failed_tracks", "failure_reasons",
                                           "replacements_used", "final_counts")}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
