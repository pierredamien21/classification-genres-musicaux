"""Durable per-track predictions and evidence gates for a frozen campaign."""

import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from music_genre.data.audit import sha256_file
from music_genre.evaluation.window_geometry import window_sample_geometry


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def atomic_json(path, value):
    """A killed writer leaves at most an ignored temporary file, never half a record."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=".pending-", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            # Close first: Windows cannot unlink an open temporary file.
            stream.close()
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def validate_prediction(prediction, labels, model_id):
    if (prediction.get("schema_version") != 1 or prediction.get("mode") != "fma"
            or prediction.get("model_id") != model_id or prediction.get("labels") != labels):
        raise ValueError("Prediction identity or labels differ")
    windows = prediction["windows"]
    geometry = window_sample_geometry(windows, 16000)
    if (len(windows) != 3 or [w[0] for w in geometry] != [0, 160000, 320000]
            or any(w[1] + w[2] != 160000 for w in geometry)):
        raise ValueError("Unexpected window geometry")
    # The geometry helper checks that times represent exact nonnegative samples.
    if len(geometry) != 3:
        raise ValueError("Unexpected window count")
    scores = np.asarray(prediction["scores"], dtype=float)
    probabilities = np.asarray([w["scores"] for w in windows], dtype=float)
    if (scores.shape != (len(labels),) or probabilities.shape != (3, len(labels))
            or not np.isfinite(scores).all() or not np.isfinite(probabilities).all()
            or np.any(probabilities < 0) or np.any(probabilities > 1)
            or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-6, rtol=0)
            or not np.allclose(scores, probabilities.mean(axis=0), atol=1e-15, rtol=0)):
        raise ValueError("Invalid probability vectors or window mean")


class PredictionStore:
    """Use under an exclusive_campaign lease; resumption must keep exact provenance."""

    def __init__(self, folder, campaign):
        self.folder = Path(folder)
        self.campaign = campaign
        self.identity = json_digest(campaign)
        path = self.folder / "campaign.json"
        if path.exists():
            if json.loads(path.read_text("utf-8")) != campaign:
                raise ValueError("Campaign differs; refusing incompatible resume")
        else:
            startup_files = {".writer.lock", "progress.json", "events.jsonl", "errors.log"}
            if self.folder.exists() and any(
                p.name not in startup_files and not (p.is_file() and p.name.startswith(".pending-")
                                                     and p.suffix == ".tmp")
                for p in self.folder.iterdir()
            ):
                raise ValueError("Unidentified nonempty campaign directory")
            atomic_json(path, campaign)

    def path(self, system, track):
        if system not in self.campaign["models"] or track not in self.campaign["track_ids"]:
            raise ValueError("Unknown system or track")
        if (not track.isdigit() or not system.replace("_", "").isalnum()
                or "/" in system or "\\" in system):
            raise ValueError("Unsafe prediction path")
        return self.folder / "predictions" / system / f"{track}.json"

    def read(self, system, track):
        path = self.path(system, track)
        if not path.exists():
            return None
        record = json.loads(path.read_text("utf-8"))
        payload = record["payload"]
        if (record["sha256"] != json_digest(payload) or payload["campaign"] != self.identity
                or payload["system"] != system or payload["track_id"] != track
                or payload["source_sha256"] != self.campaign["source_sha256"][track]):
            raise ValueError("Prediction record corrupted or incompatible")
        prediction = payload["prediction"]
        validate_prediction(prediction, self.campaign["labels"], self.campaign["models"][system])
        return prediction

    def save(self, system, track, prediction):
        path = self.path(system, track)
        validate_prediction(prediction, self.campaign["labels"], self.campaign["models"][system])
        if path.exists():
            raise FileExistsError("Completed prediction must not be overwritten")
        payload = {"campaign": self.identity, "system": system, "track_id": track,
                   "source_sha256": self.campaign["source_sha256"][track], "prediction": prediction}
        atomic_json(path, {"payload": payload, "sha256": json_digest(payload)})

    def receipts(self):
        receipts = {}
        for system in self.campaign["models"]:
            for track in self.campaign["track_ids"]:
                if self.read(system, track) is None:
                    raise ValueError("Incomplete prediction cohort")
                path = self.path(system, track)
                receipts[path.relative_to(self.folder).as_posix()] = sha256_file(path)
        actual = {p.relative_to(self.folder).as_posix()
                  for p in (self.folder / "predictions").rglob("*.json")}
        if actual != set(receipts):
            raise ValueError("Unexpected prediction files")
        return receipts


def compare_validation(store, system, reference):
    """Match historical probabilities by ID, never by row position."""
    by_id = {r["track_id"]: r for r in reference}
    ids, labels = store.campaign["track_ids"], store.campaign["labels"]
    if len(reference) != len(by_id) or set(by_id) != set(ids):
        raise ValueError("Validation reference cohort differs")
    largest, disagreements = 0., 0
    for track in ids:
        prediction = store.read(system, track)
        if prediction is None:
            raise ValueError("Incomplete validation cohort")
        actual = np.asarray(prediction["scores"])
        expected = np.array([float(by_id[track][f"p_{label}"]) for label in labels])
        if (not np.isfinite(expected).all() or np.any(expected < 0)
                or abs(expected.sum() - 1) > 1e-6):
            raise ValueError("Invalid reference probabilities")
        largest = max(largest, float(np.max(np.abs(actual - expected))))
        disagreements += int(actual.argmax() != expected.argmax())
    if largest > 1e-6 or disagreements:
        raise ValueError(f"Validation comparison failed: error={largest}, classes={disagreements}")
    return {"compared_tracks": len(ids), "max_probability_error": largest,
            "class_disagreements": disagreements}


def check_validation_evidence(folder, expected_campaign):
    """Verify the complete dry run, its source identity and every stored prediction."""
    folder = Path(folder)
    if not (folder / "campaign.json").is_file() or not (folder / "completion.json").is_file():
        raise ValueError("Missing validation evidence")
    if expected_campaign["split"] != "validation":
        raise ValueError("Expected validation campaign")
    store = PredictionStore(folder, expected_campaign)
    proof = json.loads((folder / "completion.json").read_text("utf-8"))
    if (proof["status"] != "validation_passed"
            or proof["campaign_sha256"] != store.identity
            or proof["predictions"] != store.receipts()
            or set(proof["systems"]) != set(expected_campaign["models"])):
        raise ValueError("Validation evidence incomplete or changed")
    for result in proof["systems"].values():
        error = result["max_probability_error"]
        if (result["compared_tracks"] != len(expected_campaign["track_ids"])
                or not np.isfinite(error) or not 0 <= error <= 1e-6
                or result["class_disagreements"] != 0):
            raise ValueError("Validation comparison failed")
    return proof


@contextmanager
def exclusive_campaign(folder):
    """OS lock released on process death; a leftover file does not block recovery."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / ".writer.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        if os.name == "nt":
            import msvcrt
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("Campaign already running") from exc
        else:
            import fcntl
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("Campaign already running") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
