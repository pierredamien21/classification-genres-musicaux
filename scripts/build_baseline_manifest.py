"""Apply the recorded review policy and preserve a complete decision journal."""

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from music_genre.data.audit import sha256_file, write_json
from music_genre.data.review import review_rows

ROOT = Path(__file__).resolve().parents[1]


def main():
    pointer = json.loads((ROOT / "data/manifests/latest_fingerprints.json").read_text())
    fp_folder = (ROOT / pointer["folder"]).resolve()
    if not fp_folder.is_relative_to((ROOT / "data/manifests").resolve()):
        raise ValueError("Fingerprint folder outside manifests")
    if sha256_file(fp_folder / "report.json") != pointer["report_sha256"]:
        raise ValueError("Fingerprint report changed")
    scan = json.loads((fp_folder / "report.json").read_text(encoding="utf-8"))
    audit_folder = (ROOT / scan["audit_folder"]).resolve()
    if not audit_folder.is_relative_to((ROOT / "data/manifests").resolve()):
        raise ValueError("Audit folder outside manifests")
    if sha256_file(audit_folder / "manifest.csv") != scan["audit_sha256"]:
        raise ValueError("Audit manifest changed")
    if sha256_file(fp_folder / "records.jsonl") != scan["records_sha256"]:
        raise ValueError("Fingerprint records changed")
    if sha256_file(ROOT / "configs/review_policy.json") != scan["policy_sha256"]:
        raise ValueError("Policy changed since scan; rerun with a consistent policy")
    with (audit_folder / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    records = [json.loads(line) for line in
               (fp_folder / "records.jsonl").read_text(encoding="utf-8").splitlines()]
    if len({r["track_id"] for r in records}) != len(records):
        raise ValueError("Duplicate fingerprint IDs")
    reviewed = review_rows(rows, {r["track_id"]: r for r in records}, scan["candidates"],
                           scan["policy"]["fingerprint"])
    selected = [r for r in reviewed if r["decision"] == "keep"]
    if not selected:
        raise ValueError("No eligible tracks")
    if len({r["pcm_sha256"] for r in selected}) != len(selected):
        raise ValueError("Exact PCM duplicate remains in selection")
    for key in ("artist_id", "content_group"):
        groups = {}
        for row in selected:
            if not row[key]:
                raise ValueError(f"Missing required grouping key: {key}")
            groups.setdefault(row[key], set()).add(row["split"])
        if any(len(splits) > 1 for splits in groups.values()):
            raise ValueError(f"Cross-split {key} leakage")
    folder = ROOT / "data/manifests" / ("baseline-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    folder.mkdir(exist_ok=False)
    for name, output_rows in (("decisions.csv", reviewed), ("manifest.csv", selected)):
        with (folder / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(reviewed[0]))
            writer.writeheader()
            writer.writerows(output_rows)
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(),
        "profile": scan["policy"]["name"], "audit_folder": scan["audit_folder"],
        "audit_sha256": scan["audit_sha256"], "fingerprint_folder": pointer["folder"],
        "fingerprint_report_sha256": pointer["report_sha256"],
        "policy": scan["policy"], "policy_sha256": scan["policy_sha256"],
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha256_file(p) for p in
                          [Path(__file__).resolve(), ROOT / "src/music_genre/data/review.py"]},
        "manifest_sha256": sha256_file(folder / "manifest.csv"),
        "decisions_sha256": sha256_file(folder / "decisions.csv"),
        "decision_counts": dict(Counter(r["decision"] for r in reviewed)),
        "reason_counts": dict(Counter(reason for r in reviewed
                                      for reason in r["decision_reasons"].split("|"))),
        "selected_by_split": dict(Counter(r["split"] for r in selected)),
        "selected_by_genre_and_split": {
            genre: dict(Counter(r["split"] for r in selected if r["genre"] == genre))
            for genre in sorted({r["genre"] for r in reviewed})},
        "limitations": [
            "Technical policy applied automatically; no human listening review.",
            "Hold is reversible quarantine, not adjudicated corruption.",
            "Filtered FMA-small protocol; scores not directly comparable with unfiltered benchmarks.",
            "Fingerprint retrieval is heuristic and cannot certify absence of leakage.",
            "License strings retained; no redistribution permission inferred.",
            "No model trained and no test-set performance used to select rows.",
        ],
    }
    write_json(folder / "report.json", report)
    write_json(ROOT / "data/manifests/latest_baseline.json",
               {"folder": folder.relative_to(ROOT).as_posix(),
                "manifest_sha256": report["manifest_sha256"],
                "report_sha256": sha256_file(folder / "report.json")})
    print(json.dumps({key: report[key] for key in
                      ("decision_counts", "selected_by_split", "reason_counts")}, indent=2))
    print(f"Baseline manifest: {folder / 'manifest.csv'}")


if __name__ == "__main__":
    main()
