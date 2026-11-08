"""Match extracted FMA-small files to official metadata, without decoding audio."""

import csv
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    metadata_path = ROOT / "data/raw/fma_metadata/tracks.csv"
    audio_dir = ROOT / "data/raw/fma_small"
    with metadata_path.open(encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream)
        parents, names = next(reader), next(reader)
        next(reader)  # Index-name row in the original pandas CSV export.
        columns = dict(zip(zip(parents, names), range(len(parents))))
        selected = [r for r in reader if r[columns[("set", "subset")]] == "small"]

    ids = [int(r[0]) for r in selected]
    genres = Counter(r[columns[("track", "genre_top")]] for r in selected)
    splits = Counter(r[columns[("set", "split")]] for r in selected)
    labels = json.loads((ROOT / "configs/labels.json").read_text(encoding="utf-8"))
    if len(ids) != 8000 or len(set(ids)) != 8000:
        raise RuntimeError("Expected exactly 8000 unique track IDs in FMA-small")
    if set(genres) != set(labels["labels"]) or set(genres.values()) != {1000}:
        raise RuntimeError("Metadata classes differ from the planned FMA-small dataset")

    paths = sorted(audio_dir.rglob("*.mp3"))
    audio_ids = [int(path.stem) for path in paths]
    missing = sorted(set(ids) - set(audio_ids))
    unexpected = sorted(set(audio_ids) - set(ids))
    if missing or unexpected or len(audio_ids) != len(set(audio_ids)):
        raise RuntimeError(
            f"Audio mismatch: missing={len(missing)}, unexpected={len(unexpected)}, "
            f"duplicate IDs={len(audio_ids) - len(set(audio_ids))}"
        )
    if len(paths) != 8000:
        raise RuntimeError(f"Expected 8000 MP3 files, found {len(paths)}")

    artists = {}
    for row in selected:
        split = row[columns[("set", "split")]]
        artists.setdefault(split, set()).add(row[columns[("artist", "id")]])
    overlaps = {
        f"{a}__{b}": len(artists[a] & artists[b])
        for a, b in (("training", "validation"), ("training", "test"),
                     ("validation", "test"))
    }
    report = {
        "schema_version": 1,
        "checked_at_utc": datetime.now(UTC).isoformat(),
        "source": metadata_path.relative_to(ROOT).as_posix(),
        "subset": "small",
        "tracks": len(ids),
        "unique_track_ids": len(set(ids)),
        "mp3_files": len(paths),
        "audio_bytes": sum(p.stat().st_size for p in paths),
        "genres": dict(sorted(genres.items())),
        "splits": dict(splits),
        "unique_artist_ids": len(set.union(*artists.values())),
        "artist_id_overlap_between_official_splits": overlaps,
        "audio_file_matching": "passed",
        "missing_audio_files": 0,
        "unexpected_audio_files": 0,
        "audio_decoding": "not_performed",
        "audio_content_deduplication": "not_performed",
    }
    report_path = ROOT / "data/manifests/fma_small_inventory.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
