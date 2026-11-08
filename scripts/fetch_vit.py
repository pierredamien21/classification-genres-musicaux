"""Download only the pinned checkpoint files and verify their expected hashes."""

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    specification = ROOT / "configs/vit_checkpoint.json"
    provenance = json.loads(specification.read_text(encoding="utf-8"))
    directory = ROOT / "artifacts/pretrained/vit_tiny"
    if not args.verify_only:
        snapshot_download(
            repo_id=provenance["repo_id"],
            revision=provenance["revision"],
            allow_patterns=list(provenance["sha256"]),
            local_dir=directory,
        )
    for name, expected in provenance["sha256"].items():
        if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Checkpoint checksum mismatch: {name}")
    (directory / "provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Verified pinned checkpoint: {provenance['revision']}")


if __name__ == "__main__":
    main()
