"""Recheck failed decodes and compare their source bytes with the acquired archive."""

import csv
import hashlib
import json
import zipfile
from pathlib import Path

from music_genre.data.audit import audit_audio, sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    pointer = json.loads((ROOT / "data/manifests/latest_audit.json").read_text(encoding="utf-8"))
    folder = (ROOT / pointer["folder"]).resolve()
    if not folder.is_relative_to((ROOT / "data/manifests").resolve()):
        raise ValueError("Audit folder outside manifests")
    manifest = folder / "manifest.csv"
    if sha256_file(manifest) != pointer["manifest_sha256"]:
        raise ValueError("Manifest checksum mismatch")
    with manifest.open(encoding="utf-8", newline="") as f:
        failed = [r for r in csv.DictReader(f) if r["decode_status"] == "failed"]
    settings = json.loads((folder / "settings.json").read_text(encoding="utf-8"))
    checks = []
    raw_root = (ROOT / "data/raw").resolve()
    with zipfile.ZipFile(raw_root / "archives/fma_small.zip") as archive:
        for row in failed:
            path = (raw_root / row["audio_path"]).resolve()
            if not path.is_relative_to(raw_root):
                raise ValueError("Audio path outside raw data")
            with archive.open(row["audio_path"]) as stream:
                archive_digest = hashlib.file_digest(stream, "sha256").hexdigest()
            repeated = audit_audio(path, settings)
            check = {
                "track_id": row["track_id"], "exists": path.is_file(),
                "bytes": path.stat().st_size if path.is_file() else None,
                "archive_sha256": archive_digest,
                "matches_archive_and_first_audit": (
                    archive_digest == repeated["source_sha256"] == row["source_sha256"]
                ),
                "repeat_decode_status": repeated["decode_status"],
                "repeat_error": repeated["error"],
            }
            checks.append(check)
    write_json(folder / "failure_recheck.json",
               {"manifest_sha256": pointer["manifest_sha256"], "checks": checks,
                "note": "Same decoder, independent serial retry; not a second decoder."})
    print(json.dumps(checks, indent=2))
    if any(not r["matches_archive_and_first_audit"] or r["repeat_decode_status"] != "failed"
           for r in checks):
        raise SystemExit("A failure was not reproduced; review the initial audit")


if __name__ == "__main__":
    main()
