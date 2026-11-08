"""Streaming technical audit, without resampling or genre-based decisions."""

import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def audit_audio(path: Path, settings: dict) -> dict:
    result = {
        "source_sha256": "", "pcm_sha256": "", "decode_status": "failed",
        "error": "", "sample_rate": 0, "channels": 0, "declared_frames": 0,
        "decoded_frames": 0, "duration_seconds": 0.0, "peak_abs": 0.0,
        "rms_dbfs": None, "silent_frame_fraction": 0.0,
        "near_full_scale_fraction": 0.0, "flags": [],
    }
    try:
        result["source_sha256"] = sha256_file(path)
        with sf.SoundFile(path) as source:
            rate, channels = source.samplerate, source.channels
            result.update(sample_rate=rate, channels=channels, declared_frames=source.frames)
            if rate <= 0 or channels > settings["max_channels"]:
                raise ValueError("Unsupported audio dimensions")
            if source.frames > settings["max_seconds"] * rate:
                raise ValueError("Declared duration exceeds audit limit")
            digest = hashlib.sha256(f"float32-le;sr={rate};ch={channels};".encode())
            frames = silent_frames = near_full = 0
            square_sum = peak = 0.0
            while True:
                block = source.read(rate, dtype="float32", always_2d=True)
                if not len(block):
                    break
                frames += len(block)
                if frames > settings["max_seconds"] * rate:
                    raise ValueError("Decoded duration exceeds audit limit")
                if not np.isfinite(block).all():
                    raise ValueError("Non-finite decoded samples")
                digest.update(block.astype("<f4", copy=False).tobytes(order="C"))
                squared = float(np.square(block, dtype=np.float64).sum())
                square_sum += squared
                if squared / block.size <= 10 ** (settings["silence_rms_dbfs"] / 10):
                    silent_frames += len(block)
                absolute = np.abs(block)
                peak = max(peak, float(absolute.max()))
                near_full += int(np.count_nonzero(
                    absolute >= settings["near_full_scale_threshold"]
                ))
            if not frames:
                raise ValueError("Empty decoded signal")
            result.update(
                decode_status="ok", pcm_sha256=digest.hexdigest(), decoded_frames=frames,
                duration_seconds=frames / rate, peak_abs=peak,
                rms_dbfs=10 * np.log10(square_sum / (frames * channels)) if square_sum else None,
                silent_frame_fraction=silent_frames / frames,
                near_full_scale_fraction=near_full / (frames * channels),
            )
            if frames != source.frames:
                result["flags"].append("frame_count_mismatch")
            if not settings["duration_review_min"] <= frames / rate <= settings["duration_review_max"]:
                result["flags"].append("unusual_duration")
            if not square_sum:
                result["flags"].append("all_zero_signal")
            elif result["silent_frame_fraction"] >= settings["silence_review_fraction"]:
                result["flags"].append("mostly_silent")
            if result["near_full_scale_fraction"] >= settings["near_full_scale_review_fraction"]:
                result["flags"].append("near_full_scale")
    except (OSError, ValueError, RuntimeError) as error:
        result["error"] = f"{type(error).__name__}: {error}"
    return result


def duplicate_groups(rows: list[dict], field: str) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        if row.get(field):
            groups[row[field]].append(row)
    return [
        {"sha256": key, "track_ids": sorted(r["track_id"] for r in group),
         "splits": sorted({r["split"] for r in group}),
         "cross_split": len({r["split"] for r in group}) > 1}
        for key, group in sorted(groups.items()) if len(group) > 1
    ]


def parse_decoder_log(text: str) -> dict[str, str]:
    """Associate native stderr with sequential file markers in a dedicated worker."""
    current = None
    messages = defaultdict(list)
    for line in text.splitlines():
        if line.startswith("AUDIT_BEGIN "):
            if current is not None:
                raise ValueError("Nested decoder log markers")
            current = line.split()[1]
        elif line.startswith("AUDIT_END "):
            if current != line.split()[1]:
                raise ValueError("Mismatched decoder log markers")
            current = None
        elif line.strip():
            if current is None:
                raise ValueError(f"Unattributed decoder diagnostic: {line}")
            messages[current].append(line)
    if current is not None:
        raise ValueError("Incomplete decoder log")
    return {key: "\n".join(lines) for key, lines in messages.items()}


def classify_row(row: dict) -> None:
    """Exclude only unusable signals; keep all other flagged data pending review."""
    if row["decode_status"] != "ok":
        row.update(status="excluded", exclusion_reason="decode_failed")
    elif "all_zero_signal" in row["flags"]:
        row.update(status="excluded", exclusion_reason="all_zero_signal")
    else:
        row.update(status="needs_review" if row["flags"] else "valid", exclusion_reason="")
    row["flags"] = sorted(set(row["flags"]))


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")
