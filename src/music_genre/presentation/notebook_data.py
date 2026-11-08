"""Small evidence boundary shared by the pedagogical notebooks."""

import hashlib
import json
from pathlib import Path


def read_verified_json(root: Path, relative_path: str, expected_sha256: str):
    """Read a JSON artifact only if its bytes match the recorded digest."""
    path = root / relative_path
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError(f"Empreinte différente : {relative_path}. Vérifier la provenance.")
    return json.loads(raw)
