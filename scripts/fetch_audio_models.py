"""Download the pinned audio-pretrained checkpoints (configs/audio_models.json) and verify them.

Weights are checked against the published SHA-256 (Hugging Face LFS) or MD5 (Zenodo); the
SHA-256 of every file is then recorded in artifacts/pretrained/<model>/provenance.json.
No downloaded code is executed.

  python scripts/fetch_audio_models.py [--model ast] [--verify-only]
"""

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]


def digest(path, algorithm):
    hasher = hashlib.new(algorithm)
    with path.open("rb") as stream:
        while block := stream.read(1 << 20):
            hasher.update(block)
    return hasher.hexdigest()


def fetch(name, spec, verify_only):
    directory = ROOT / "artifacts/pretrained" / name
    directory.mkdir(parents=True, exist_ok=True)
    if not verify_only:
        if spec["source"] == "huggingface":
            for filename in spec["files"]:
                hf_hub_download(spec["repo_id"], filename, revision=spec["revision"], local_dir=directory)
        else:
            (filename,) = spec["files"]
            target = directory / filename
            if not (target.exists() and target.stat().st_size == spec["bytes"][filename]):
                partial = target.with_suffix(".partial")
                request = urllib.request.Request(spec["url"], headers={"User-Agent": "music-genre-v2-fetch"})
                with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as stream:
                    while block := response.read(1 << 20):
                        stream.write(block)
                partial.replace(target)
    for filename, expected in spec.get("weights_sha256", {}).items():
        if digest(directory / filename, "sha256") != expected:
            raise ValueError(f"{name}: SHA-256 mismatch for {filename}")
    for filename, expected in spec.get("weights_md5", {}).items():
        if digest(directory / filename, "md5") != expected:
            raise ValueError(f"{name}: MD5 mismatch for {filename}")
    provenance = {**spec, "sha256": {f: digest(directory / f, "sha256") for f in spec["files"]}}
    (directory / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"{name}: verified {len(spec['files'])} files")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    models = json.loads((ROOT / "configs/audio_models.json").read_text(encoding="utf-8"))["models"]
    for name, spec in models.items():
        if args.model in (None, name):
            fetch(name, spec, args.verify_only)


if __name__ == "__main__":
    main()
