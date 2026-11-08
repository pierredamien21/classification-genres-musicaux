"""List V2 tracks whose decoding (libsndfile/libmpg123, via audit_audio) writes to stderr.

Not an exclusion rule: the pre-registered audit ignores these messages. The list feeds a
descriptive sensitivity analysis declared before any V2 model run (docs/PROTOCOLE_V2.md).
Decodes sequentially so each message is attributed to one file.

Writes reports/v2/decoder_messages.csv.
"""

import csv
import json
import os
import re
import tempfile
from pathlib import Path

from music_genre.data.audit import audit_audio

ROOT = Path(__file__).resolve().parents[1]
COHORTS = ("reports/v2/cohorts.csv", "reports/v2/cohorts_amendment.csv")


def main():
    settings = json.loads((ROOT / "configs/audit.json").read_text())
    rows = []
    for name in COHORTS:
        with (ROOT / name).open(encoding="utf-8", newline="") as stream:
            rows += list(csv.DictReader(stream))
    with (ROOT / "reports/v2/final_cohorts.csv").open(encoding="utf-8", newline="") as stream:
        final = {r["track_id"] for r in csv.DictReader(stream)}
    found = []
    saved = os.dup(2)
    with tempfile.TemporaryFile() as capture:
        os.dup2(capture.fileno(), 2)  # C libraries write straight to file descriptor 2
        try:
            for row in rows:
                capture.seek(0)
                capture.truncate()
                audit_audio(ROOT / "data/raw/fma_large" / Path(row["zip_member"]).relative_to("fma_large"), settings)
                capture.seek(0)
                lines = [line for line in capture.read().decode("utf-8", "replace").splitlines() if line.strip()]
                if lines:
                    kinds = sorted({re.sub(r"\d+", "N", re.sub(r"^\[.*?\]\s*", "", line)) for line in lines})
                    found.append({"track_id": row["track_id"], "role": row["role"], "genre": row["genre"],
                                  "in_final_cohorts": row["track_id"] in final, "lines": len(lines),
                                  "messages": " | ".join(kinds)})
        finally:
            os.dup2(saved, 2)
    with (ROOT / "reports/v2/decoder_messages.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["track_id", "role", "genre", "in_final_cohorts", "lines",
                                                    "messages"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(found)
    print(f"{len(found)} files with decoder messages, {sum(r['in_final_cohorts'] for r in found)} in final cohorts")


if __name__ == "__main__":
    main()
