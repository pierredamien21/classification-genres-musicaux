"""Amendment 1 to the V2 protocol: extra Pop spares, drawn from metadata only.

The pre-registered Pop spares ran out (45 of 80 Pop tracks are 0-15 s clips in fma_large).
Every remaining eligible Pop artist (medium/large, absent from FMA-small and from every V2
cohort) with at least one track whose metadata duration is >= 30 s is ranked with a new seed;
one track >= 30 s is drawn per artist with the same pick() as the original selection.
Ranks continue after the original Pop spares, so audit_v2_cohorts.replace() uses them last.
Decided before any V2 model run; uses metadata only (no audio, model or prediction).

Writes reports/v2/cohorts_amendment.csv and reports/v2/amendment.json.
"""

import csv
import importlib.util
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from music_genre.data.audit import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261109
MIN_SECONDS = 30
spec = importlib.util.spec_from_file_location("select_v2_cohorts", ROOT / "scripts/select_v2_cohorts.py")
select = importlib.util.module_from_spec(spec)
spec.loader.exec_module(select)


def main():
    metadata = ROOT / "data/raw/fma_metadata/tracks.csv"
    tracks = pd.read_csv(metadata, index_col=0, header=[0, 1], low_memory=False)
    frame = pd.DataFrame({"subset": tracks[("set", "subset")], "genre": tracks[("track", "genre_top")],
                          "artist": tracks[("artist", "id")], "duration": tracks[("track", "duration")]})
    small_artists = set(frame.artist[frame.subset == "small"])
    with (ROOT / "reports/v2/cohorts.csv").open(encoding="utf-8", newline="") as stream:
        cohorts = list(csv.DictReader(stream))
    used = {int(r["artist_id"]) for r in cohorts}
    pool = frame[frame.subset.isin(["medium", "large"]) & ~frame.artist.isin(small_artists)
                 & (frame.genre == "Pop") & (frame.duration >= MIN_SECONDS)]
    eligible = pool.artist[~pool.artist.isin(used)].nunique()
    rng = np.random.Generator(np.random.PCG64(SEED))
    rows = select.pick(pool, ["Pop"], set(used), rng, "test", 0, eligible)
    offset = 1 + max(int(r["rank"]) for r in cohorts if r["genre"] == "Pop" and r["role"].startswith("test"))
    for row in rows:
        row["rank"] += offset
        row["subset"] = tracks.loc[int(row["track_id"]), ("set", "subset")]
        row["zip_member"] = f"fma_large/{row['track_id'][:3]}/{row['track_id']}.mp3"
    # Self-checks: new artists only, one track each, all spares, all >= 30 s by metadata.
    artists = {r["artist_id"] for r in rows}
    assert len(artists) == len(rows) == eligible and not artists & (used | small_artists)
    assert all(r["role"] == "test_spare" for r in rows)
    assert all(tracks.loc[int(r["track_id"]), ("track", "duration")] >= MIN_SECONDS for r in rows)

    out = ROOT / "reports/v2/cohorts_amendment.csv"
    with out.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(cohorts[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(ROOT / "reports/v2/amendment.json", {
        "schema_version": 1, "amendment": 1, "created_at_utc": datetime.now(UTC).isoformat(),
        "reason": "Pop spares exhausted by 0-15 s clips; Pop test kept 35 of 64 tracks",
        "seed": SEED, "min_metadata_seconds": MIN_SECONDS, "first_rank": offset, "tracks": len(rows),
        "metadata_sha256": sha256_file(metadata), "cohorts_sha256": sha256_file(ROOT / "reports/v2/cohorts.csv"),
        "script_sha256": sha256_file(Path(__file__)), "amendment_sha256": sha256_file(out),
        "audio_loaded": False, "models_run": False, "predictions_loaded": False})
    print(f"{len(rows)} Pop spares, ranks {offset}-{offset + len(rows) - 1}")


if __name__ == "__main__":
    main()
