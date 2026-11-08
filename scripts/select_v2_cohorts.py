"""Pre-register the V2 cohorts from FMA metadata only (no audio, no predictions).

- test: per in-domain genre, artists never seen in FMA-small, one track each, 64 in seeded
  order, then spares (replacements, used in order only after a failed technical audit).
- ood: same rule for the FMA genres outside the eight classes; gospel_probe: tracks tagged
  Gospel (genre 567). Artists are disjoint across every role and from FMA-small.
- dev_folds: 5 artist-grouped, genre-stratified folds over the selected FMA-small tracks.

Writes reports/v2/cohorts.csv, reports/v2/dev_folds.csv and reports/v2/selection.json.
"""

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from music_genre.data.audit import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261108
PER_GENRE = 64
SPARES = 16
OOD_GENRES = ("Blues", "Classical", "Country", "Easy Listening", "Jazz", "Old-Time / Historic",
              "Soul-RnB", "Spoken")
GOSPEL = 567


def pick(pool, genres, used, rng, role, per_genre, spares):
    """Seeded artist order per genre, one uniformly drawn track per artist."""
    rows = []
    # Scarcest genre first, so a multi-genre artist is not consumed by an abundant genre.
    for genre in sorted(genres, key=lambda g: pool[pool.genre == g].artist.nunique()):
        tracks = pool[(pool.genre == genre) & ~pool.artist.isin(used)]
        artists = sorted(tracks.artist.unique())
        for rank, artist in enumerate(rng.permutation(artists)[:per_genre + spares]):
            choices = sorted(tracks[tracks.artist == artist].index)
            rows.append({"role": role if rank < per_genre else role + "_spare", "rank": rank,
                         "genre": genre, "artist_id": int(artist),
                         "track_id": f"{int(rng.choice(choices)):06d}"})
            used.add(artist)
    return rows


def main():
    metadata = ROOT / "data/raw/fma_metadata/tracks.csv"
    labels = json.loads((ROOT / "configs/labels.json").read_text())["labels"]
    tracks = pd.read_csv(metadata, index_col=0, header=[0, 1], low_memory=False)
    frame = pd.DataFrame({"subset": tracks[("set", "subset")], "genre": tracks[("track", "genre_top")],
                          "artist": tracks[("artist", "id")],
                          "genres_all": tracks[("track", "genres_all")].astype(str)})
    small_artists = set(frame.artist[frame.subset == "small"])
    pool = frame[frame.subset.isin(["medium", "large"]) & ~frame.artist.isin(small_artists)]
    rng = np.random.Generator(np.random.PCG64(SEED))
    used = set()
    rows = pick(pool, labels, used, rng, "test", PER_GENRE, SPARES)
    rows += pick(pool, OOD_GENRES, used, rng, "ood", PER_GENRE, SPARES)
    gospel = pool[pool.genres_all.str.contains(rf"\b{GOSPEL}\b") & ~pool.artist.isin(used)].copy()
    gospel["genre"] = "Gospel"
    rows += pick(gospel, ["Gospel"], used, rng, "gospel_probe", PER_GENRE, 0)
    for row in rows:
        row["subset"] = tracks.loc[int(row["track_id"]), ("set", "subset")]
        row["zip_member"] = f"fma_large/{row['track_id'][:3]}/{row['track_id']}.mp3"
    # Self-checks: one track per artist, no FMA-small artist, no duplicate track.
    assert len({r["artist_id"] for r in rows}) == len(rows)
    assert not {r["artist_id"] for r in rows} & small_artists
    assert len({r["track_id"] for r in rows}) == len(rows)

    baseline = json.loads((ROOT / "data/manifests/latest_baseline.json").read_text())
    with (ROOT / baseline["folder"] / "manifest.csv").open(encoding="utf-8", newline="") as stream:
        dev = sorted((r for r in csv.DictReader(stream) if r["decision"] == "keep"),
                     key=lambda r: r["track_id"])
    folds = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    assignment = {}
    for fold, (_, held) in enumerate(folds.split(dev, [r["genre"] for r in dev], [r["artist_id"] for r in dev])):
        for i in held:
            assignment[dev[i]["track_id"]] = fold
    by_artist = {}
    for r in dev:
        by_artist.setdefault(r["artist_id"], set()).add(assignment[r["track_id"]])
    assert len(assignment) == len(dev) and all(len(f) == 1 for f in by_artist.values())

    out = ROOT / "reports/v2"
    out.mkdir(parents=True, exist_ok=True)
    fields = ["role", "rank", "genre", "artist_id", "track_id", "subset", "zip_member"]
    with (out / "cohorts.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with (out / "dev_folds.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["track_id", "artist_id", "genre", "v1_split", "fold"])
        writer.writerows([r["track_id"], r["artist_id"], r["genre"], r["split"], assignment[r["track_id"]]]
                         for r in dev)
    counts = Counter((r["role"], r["genre"]) for r in rows)
    write_json(out / "selection.json", {
        "schema_version": 1, "created_at_utc": datetime.now(UTC).isoformat(), "seed": SEED,
        "per_genre": PER_GENRE, "spares_per_genre": SPARES, "ood_genres": list(OOD_GENRES),
        "gospel_genre_id": GOSPEL, "metadata_sha256": sha256_file(metadata),
        "dev_manifest_sha256": baseline["manifest_sha256"],
        "script_sha256": sha256_file(Path(__file__)),
        "counts": {f"{role}/{genre}": n for (role, genre), n in sorted(counts.items())},
        "dev_tracks": len(dev), "dev_fold_sizes": dict(sorted(Counter(assignment.values()).items())),
        "audio_loaded": False, "predictions_loaded": False,
        "cohorts_sha256": sha256_file(out / "cohorts.csv"),
        "dev_folds_sha256": sha256_file(out / "dev_folds.csv"),
    })
    print(json.dumps({f"{r}/{g}": n for (r, g), n in sorted(counts.items())}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
