"""Numeric per-track checkpoints with identities and atomic completion markers."""

import json

import numpy as np

from music_genre.data.audit import sha256_file


def _validate(x, valid):
    if (x.shape != (3, 192) or valid.shape != (3,) or not np.isfinite(x).all()
            or not np.isfinite(valid).all() or np.any(valid <= 0) or np.any(valid > 10)):
        raise ValueError("Invalid three-window track cache")


def save_track_cache(path, cache_key, source_sha256, x, valid):
    _validate(x, valid)
    temporary = path.with_suffix(".partial.npz")
    np.savez(temporary, cache_key=cache_key, source_sha256=source_sha256, x=x, valid=valid)
    temporary.replace(path)
    marker = path.with_suffix(".partial.json")
    marker.write_text(json.dumps({"sha256": sha256_file(path)}), encoding="utf-8")
    marker.replace(path.with_suffix(".json"))


def load_track_cache(path, cache_key, source_sha256):
    marker = path.with_suffix(".json")
    if not path.exists() or not marker.exists():
        return None
    if json.loads(marker.read_text(encoding="utf-8"))["sha256"] != sha256_file(path):
        raise ValueError("Track cache checksum mismatch")
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["cache_key"]) != cache_key or str(saved["source_sha256"]) != source_sha256:
            raise ValueError("Stale track cache")
        x, valid = saved["x"].copy(), saved["valid"].copy()
    _validate(x, valid)
    return x, valid
