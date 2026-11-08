"""Small positive/negative controls from training audio; not a global recall estimate."""

import csv
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import soundfile as sf

from music_genre.audio.processing import load_audio
from music_genre.data.audit import sha256_file, write_json
from music_genre.data.fingerprints import find_candidates

ROOT = Path(__file__).resolve().parents[1]


def main():
    policy_path = ROOT / "configs/review_policy.json"
    settings = json.loads(policy_path.read_text())["fingerprint"]
    executable = ROOT / settings["executable"]
    if sha256_file(executable) != settings["executable_sha256"]:
        raise ValueError("Unexpected executable hash")
    pointer = json.loads((ROOT / "data/manifests/latest_audit.json").read_text())
    manifest = ROOT / pointer["folder"] / "manifest.csv"
    if sha256_file(manifest) != pointer["manifest_sha256"]:
        raise ValueError("Audit changed")
    with manifest.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    chosen = [next(r for r in rows if r["split"] == "train" and r["status"] == "valid"
                   and r["genre"] == genre) for genre in ("Electronic", "Folk", "Hip-Hop")]
    directory = ROOT / "data/interim" / ("fingerprint-controls-" +
                                        datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
    directory.mkdir(exist_ok=False)
    records = []
    expected = []

    def fingerprint(path, key):
        process = subprocess.run(
            [str(executable), "-raw", "-json", "-algorithm", str(settings["algorithm"]),
             "-length", str(settings["max_seconds"]), str(path)],
            check=False, capture_output=True, timeout=settings["timeout_seconds"],
        )
        if process.returncode or process.stderr.strip():
            raise RuntimeError(f"Control decode error: {process.stderr.decode(errors='replace')}")
        return {"track_id": key, "returncode": 0,
                "fingerprint": json.loads(process.stdout)["fingerprint"]}

    for row in chosen:
        key = row["track_id"]
        source = ROOT / "data/raw" / row["audio_path"]
        if sha256_file(source) != row["source_sha256"]:
            raise ValueError(f"Source changed: {key}")
        records.append(fingerprint(source, key))
        audio = load_audio(source)
        variants = {
            "mono16k_gain": audio.samples * 0.2,
            "mono16k_shift2s": np.concatenate(
                [np.zeros(2 * audio.sample_rate, dtype=np.float32),
                 audio.samples[:-2 * audio.sample_rate]]
            ),
        }
        for name, values in variants.items():
            target = directory / f"{key}_{name}.wav"
            sf.write(target, values, audio.sample_rate, subtype="FLOAT")
            variant_id = f"{key}_{name}"
            records.append(fingerprint(target, variant_id))
            expected.append(tuple(sorted((key, variant_id))))
    matches = find_candidates(records, settings)
    found = {tuple(sorted((r["a"], r["b"]))) for r in matches}
    positives = [{"a": a, "b": b, "detected": (a, b) in found} for a, b in expected]
    negatives = [r for r in matches if r["a"].split("_")[0] != r["b"].split("_")[0]]
    report = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "audit_sha256": pointer["manifest_sha256"], "policy_sha256": sha256_file(policy_path),
        "source_sha256": sha256_file(Path(__file__).resolve()),
        "training_track_ids": [r["track_id"] for r in chosen],
        "positive_controls": positives, "unexpected_cross_track_matches": negatives,
        "matches": matches,
        "limitations": [
            "Three training sources, six transformations; not a dataset-wide recall estimate.",
            "No threshold tuned using these outcomes or validation/test performance.",
            "WAV conversion, gain and 2-second shift only; no MP3 re-encoding control.",
        ],
    }
    write_json(ROOT / "reports/FINGERPRINT_CONTROLS.json", report)
    print(json.dumps({"positive_controls": positives,
                      "unexpected_cross_track_matches": negatives}, indent=2))


if __name__ == "__main__":
    main()
