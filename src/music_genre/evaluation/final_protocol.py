"""Integrity and cohort gates for a frozen scientific evaluation protocol."""

import csv
import json
import platform
from importlib.metadata import version
from pathlib import Path, PurePosixPath

from music_genre.data.audit import sha256_file


def inside(root, relative):
    if not isinstance(relative, str) or "\\" in relative:
        raise ValueError("Expected a relative POSIX artifact path")
    relative_path = PurePosixPath(relative)
    path = (root / relative).resolve()
    if relative_path.is_absolute() or ".." in relative_path.parts or not path.is_relative_to(root):
        raise ValueError("Artifact escapes workspace")
    return path


def inspect_partitions(rows, labels):
    """Check IDs, class order, artist/content disjointness and fixed audio hashes."""
    splits = {s: [] for s in ("train", "validation", "test")}
    seen, ownership = set(), {}
    if not rows or len(labels) != len(set(labels)) or len(labels) < 2:
        raise ValueError("Empty dataset or invalid labels")
    for row in rows:
        track, artist, split = row["track_id"], row["artist_id"], row["split"]
        if not track or not artist or track in seen or split not in splits:
            raise ValueError("Missing/duplicate identity or invalid split")
        label = int(row["label_index"])
        if not 0 <= label < len(labels) or row["genre"] != labels[label]:
            raise ValueError("Label order mismatch")
        seen.add(track)
        splits[split].append(track)
        for key in ("artist_id", "pcm_sha256", "content_group"):
            value = row.get(key)
            if value:
                token = (key, value)
                if token in ownership and ownership[token] != split:
                    raise ValueError(f"Cross-partition {key}")
                ownership[token] = split
    if any(not tracks for tracks in splits.values()):
        raise ValueError("Empty partition")
    return {split: sorted(tracks) for split, tracks in splits.items()}


def load_locked_manifest(root, relative, expected_sha256):
    """Verify all pinned files and environment once, before any evaluation decode.

    expected_sha256 comes from the trusted local lock record. This detects drift,
    not an attacker rewriting both the manifest and its trusted digest.
    """
    root = Path(root).resolve()
    path = inside(root, relative)
    if sha256_file(path) != expected_sha256:
        raise ValueError("Lock digest mismatch")
    locked = json.loads(path.read_text("utf-8"))
    if (locked.get("schema_version") != 1 or locked.get("status") != "locked"
            or locked.get("test_evaluation_allowed") is not True
            or locked["protocol"]["refit_on_train_and_validation"] is not False):
        raise ValueError("Protocol is not locked for evaluation")
    artifacts = locked["artifacts"]
    for name, digest in artifacts.items():
        target = inside(root, name)
        if not target.is_file() or sha256_file(target) != digest:
            raise ValueError(f"Pinned artifact missing or changed: {name}")
    if not locked["source_files"] or any(p not in artifacts for p in locked["source_files"]):
        raise ValueError("Required source is not pinned")
    names = locked["protocol"]["linear_systems"] + locked["protocol"]["cnn_runs"]
    if len(names) != len(set(names)) or set(names) != set(locked["models"]):
        raise ValueError("Model roster mismatch")
    if any(pin["model_path"] not in artifacts for pin in locked["models"].values()):
        raise ValueError("Model is not pinned")
    if set(locked["proofs"]) != {"T1", "T2", "T2a", "T2b", "T3", "T4", "T5"}:
        raise ValueError("Missing prerequisite proof")
    for name in locked["proofs"].values():
        if name not in artifacts:
            raise ValueError("Proof is not pinned")
        proof = json.loads(inside(root, name).read_text("utf-8"))
        if proof.get("status") != "complete" or proof.get("test_evaluated") is not False:
            raise ValueError("Prerequisite incomplete or test already evaluated")
    if locked["robustness_disposition"] != "accepted_limitation_after_T2a_T2b":
        raise ValueError("Robustness review unresolved")
    dataset = locked["dataset"]
    if dataset["manifest"] not in artifacts:
        raise ValueError("Dataset manifest is not pinned")
    with inside(root, dataset["manifest"]).open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    ids = inspect_partitions(rows, locked["labels"])
    if ids != dataset["track_ids"] or {s: len(v) for s, v in ids.items()} != dataset["counts"]:
        raise ValueError("Dataset cohort differs from lock")
    for row in rows:
        raw = "data/raw/" + row["audio_path"]
        if artifacts.get(raw) != row["source_sha256"]:
            raise ValueError("Raw audio is not pinned to the dataset")
    for package, expected in locked["versions"].items():
        actual = platform.python_version() if package == "python" else version(package)
        if actual != expected:
            raise ValueError(f"Runtime version changed: {package}")
    return locked


def validate_evaluation_request(locked, track_ids, split):
    """Reject subsets, duplicates, train requests and cross-split evaluation lists."""
    if locked.get("status") != "locked" or locked.get("test_evaluation_allowed") is not True:
        raise ValueError("Protocol is not locked for evaluation")
    if split not in ("validation", "test"):
        raise ValueError("Expected validation or test evaluation")
    if len(track_ids) != len(set(track_ids)) or sorted(track_ids) != locked["dataset"]["track_ids"][split]:
        raise ValueError("Evaluation IDs do not match the complete locked partition")
