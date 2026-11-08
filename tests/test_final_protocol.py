import csv
import json

import pytest

from music_genre.data.audit import sha256_file
from music_genre.evaluation.final_protocol import load_locked_manifest, validate_evaluation_request


def locked_fixture(root):
    (root / "reports").mkdir()
    (root / "data/raw").mkdir(parents=True)
    (root / "src").mkdir()
    (root / "src/predict.py").write_text("# fixture source")
    (root / "model.bin").write_bytes(b"fixed model")
    rows = []
    for i, split in enumerate(("train", "validation", "test")):
        audio = root / f"data/raw/{i}.bin"
        audio.write_bytes(str(i).encode())
        rows.append({"track_id": str(i), "artist_id": f"artist{i}", "split": split,
                     "label_index": str(i % 2), "genre": ["a", "b"][i % 2],
                     "audio_path": f"{i}.bin", "source_sha256": sha256_file(audio),
                     "pcm_sha256": sha256_file(audio), "content_group": str(i)})
    with (root / "data/manifest.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "reports/proof.json").write_text(json.dumps({"status": "complete", "test_evaluated": False}))
    paths = ["src/predict.py", "model.bin", "data/manifest.csv", "reports/proof.json"] + [f"data/raw/{i}.bin" for i in range(3)]
    manifest = {"schema_version": 1, "status": "locked", "test_evaluation_allowed": True,
                "protocol": {"linear_systems": ["model"], "cnn_runs": [], "refit_on_train_and_validation": False},
                "labels": ["a", "b"], "models": {"model": {"model_path": "model.bin"}},
                "artifacts": {p: sha256_file(root / p) for p in paths}, "source_files": ["src/predict.py"],
                "proofs": {p: "reports/proof.json" for p in ("T1", "T2", "T2a", "T2b", "T3", "T4", "T5")},
                "dataset": {"manifest": "data/manifest.csv", "counts": {s: 1 for s in ("train", "validation", "test")},
                            "track_ids": {s: [str(i)] for i, s in enumerate(("train", "validation", "test"))}},
                "versions": {}, "robustness_disposition": "accepted_limitation_after_T2a_T2b"}
    return save_lock(root, manifest), manifest


def save_lock(root, manifest):
    path = root / "reports/lock.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return sha256_file(path)


def test_locked_protocol_allows_exact_validation_dry_run(tmp_path):
    digest, _ = locked_fixture(tmp_path)
    locked = load_locked_manifest(tmp_path, "reports/lock.json", digest)
    validate_evaluation_request(locked, ["1"], "validation")
    validate_evaluation_request(locked, ["2"], "test")


@pytest.mark.parametrize("kind", ["unlocked", "not_allowed", "digest", "model_changed",
                                  "missing_source", "unpinned_source", "missing_proof",
                                  "failed_proof", "version", "outside", "roster", "review"])
def test_lock_rejects_invalid_state_or_artifacts(tmp_path, kind):
    digest, manifest = locked_fixture(tmp_path)
    if kind == "unlocked":
        manifest["status"] = "prepared_not_locked"
    elif kind == "not_allowed":
        manifest["test_evaluation_allowed"] = False
    elif kind == "digest":
        digest = "0" * 64
    elif kind == "model_changed":
        (tmp_path / "model.bin").write_bytes(b"new model")
    elif kind == "missing_source":
        (tmp_path / "src/predict.py").unlink()
    elif kind == "unpinned_source":
        del manifest["artifacts"]["src/predict.py"]
    elif kind == "missing_proof":
        del manifest["proofs"]["T1"]
    elif kind == "failed_proof":
        path = tmp_path / "reports/proof.json"
        path.write_text(json.dumps({"status": "failed", "test_evaluated": False}))
        manifest["artifacts"]["reports/proof.json"] = sha256_file(path)
    elif kind == "version":
        manifest["versions"] = {"python": "0.0"}
    elif kind == "outside":
        manifest["artifacts"]["../outside"] = "0" * 64
    elif kind == "roster":
        manifest["protocol"]["linear_systems"].append("extra")
    else:
        manifest["robustness_disposition"] = "pending"
    if kind != "digest":
        digest = save_lock(tmp_path, manifest)
    with pytest.raises(ValueError):
        load_locked_manifest(tmp_path, "reports/lock.json", digest)


@pytest.mark.parametrize("kind", ["artist", "content", "duplicate", "label", "cohort", "raw_hash"])
def test_lock_rejects_inconsistent_dataset_even_with_updated_manifest_hash(tmp_path, kind):
    _, locked = locked_fixture(tmp_path)
    path = tmp_path / "data/manifest.csv"
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if kind == "artist":
        rows[2]["artist_id"] = rows[0]["artist_id"]
    elif kind == "content":
        rows[2]["pcm_sha256"] = rows[0]["pcm_sha256"]
    elif kind == "duplicate":
        rows[2]["track_id"] = rows[0]["track_id"]
    elif kind == "label":
        rows[0]["genre"] = "b"
    elif kind == "cohort":
        locked["dataset"]["track_ids"]["test"] = ["other"]
    else:
        rows[0]["source_sha256"] = "0" * 64
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    locked["artifacts"]["data/manifest.csv"] = sha256_file(path)
    with pytest.raises(ValueError):
        load_locked_manifest(tmp_path, "reports/lock.json", save_lock(tmp_path, locked))


@pytest.mark.parametrize("ids,split", [(["0"], "train"), (["1"], "test"),
                                      (["1", "2"], "test"), (["2", "2"], "test"), ([], "test")])
def test_evaluation_request_rejects_wrong_partition_or_incomplete_ids(tmp_path, ids, split):
    digest, _ = locked_fixture(tmp_path)
    locked = load_locked_manifest(tmp_path, "reports/lock.json", digest)
    with pytest.raises(ValueError):
        validate_evaluation_request(locked, ids, split)
