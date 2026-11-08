"""Fetch the pinned official fpcalc utility, verifying archive and executable hashes."""

import json
import urllib.request
import zipfile
from pathlib import Path

from music_genre.data.audit import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def main():
    settings = json.loads((ROOT / "configs/review_policy.json").read_text())["fingerprint"]
    directory = ROOT / "data/interim/tools"
    directory.mkdir(parents=True, exist_ok=True)
    archive = directory / f"chromaprint-{settings['version']}.zip"
    if not archive.exists():
        urllib.request.urlretrieve(settings["archive_url"], archive)
    if sha256_file(archive) != settings["archive_sha256"]:
        raise ValueError("Unexpected Chromaprint archive hash")
    with zipfile.ZipFile(archive) as source:
        for member in source.infolist():
            if not (directory / member.filename).resolve().is_relative_to(directory.resolve()):
                raise ValueError("Unsafe archive path")
        source.extractall(directory)
    executable = ROOT / settings["executable"]
    if sha256_file(executable) != settings["executable_sha256"]:
        raise ValueError("Unexpected fpcalc executable hash")
    print(f"Verified: {executable}")


if __name__ == "__main__":
    main()
