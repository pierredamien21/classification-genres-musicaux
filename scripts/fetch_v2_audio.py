"""Fetch only the pre-registered V2 tracks from the official fma_large.zip with HTTPS range reads.

Reads the remote ZIP directory, then each selected member (ZIP CRC-32 checked by zipfile),
writes data/raw/fma_large/<xxx>/<track>.mp3 atomically and records size and SHA-256.
Resumable: files already present with a matching CRC-32 are kept. Standard library only.
"""

import argparse
import csv
import urllib.request
import zipfile
import zlib
from datetime import UTC, datetime
from pathlib import Path

from music_genre.data.audit import sha256_file, write_json

ROOT = Path(__file__).resolve().parents[1]

URL = "https://os.unil.cloud.switch.ch/fma/fma_large.zip"
EXPECTED_BYTES = 100_306_112_191  # Content-Length observed on 2026-11-08
BLOCK = 1 << 20


class RangeFile:
    """Read-only seekable view over fetch(start, stop) -> bytes, with a one-block cache."""

    def __init__(self, fetch, size):
        self.fetch, self.size, self.position, self.cache = fetch, size, 0, (None, b"")

    def seekable(self):
        return True

    def tell(self):
        return self.position

    def seek(self, offset, whence=0):
        self.position = {0: offset, 1: self.position + offset, 2: self.size + offset}[whence]
        return self.position

    def read(self, count=-1):
        stop = self.size if count is None or count < 0 else min(self.size, self.position + count)
        start, self.position = self.position, max(self.position, stop)
        if stop <= start:
            return b""
        cached_start, data = self.cache
        if cached_start is not None and cached_start <= start and stop <= cached_start + len(data):
            return data[start - cached_start:stop - cached_start]
        data = self.fetch(start, max(stop, min(self.size, start + BLOCK)))
        self.cache = (start, data)
        return data[:stop - start]


def https_fetch(start, stop):
    request = urllib.request.Request(URL, headers={"Range": f"bytes={start}-{stop - 1}",
                                                   "User-Agent": "music-genre-v2-fetch"})
    with urllib.request.urlopen(request, timeout=120) as response:
        if response.status != 206:
            raise RuntimeError(f"Range request refused: HTTP {response.status}")
        data = response.read()
    if len(data) != stop - start:
        raise RuntimeError("Short range read")
    return data


def crc32_file(path):
    crc = 0
    with path.open("rb") as stream:
        while block := stream.read(BLOCK):
            crc = zlib.crc32(block, crc)
    return crc


def fetch_members(archive, members, target):
    records = []
    for name in members:
        info = archive.getinfo(name)
        path = target / Path(name).relative_to("fma_large")
        if not (path.exists() and path.stat().st_size == info.file_size and crc32_file(path) == info.CRC):
            path.parent.mkdir(parents=True, exist_ok=True)
            partial = path.with_suffix(".partial")
            partial.write_bytes(archive.read(name))  # zipfile raises on CRC mismatch
            partial.replace(path)
        records.append({"zip_member": name, "bytes": info.file_size, "crc32": f"{info.CRC:08x}",
                        "sha256": sha256_file(path)})
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohorts", default="reports/v2/cohorts.csv")
    parser.add_argument("--manifest", default="data/manifests/v2_audio.json")
    args = parser.parse_args(argv)
    with (ROOT / args.cohorts).open(encoding="utf-8", newline="") as stream:
        members = [r["zip_member"] for r in csv.DictReader(stream)]
    size = int(urllib.request.urlopen(urllib.request.Request(URL, method="HEAD"), timeout=60)
               .headers["Content-Length"])
    if size != EXPECTED_BYTES:
        raise RuntimeError(f"Remote archive size changed: {size}")
    with zipfile.ZipFile(RangeFile(https_fetch, size)) as archive:
        missing = [m for m in members if m not in archive.NameToInfo]
        if missing:
            raise RuntimeError(f"{len(missing)} selected members absent from the archive: {missing[:5]}")
        print(f"{len(members)} members, {sum(archive.getinfo(m).file_size for m in members) / 1e9:.2f} GB",
              flush=True)
        records = fetch_members(archive, members, ROOT / "data/raw/fma_large")
    write_json(ROOT / args.manifest, {
        "schema_version": 1, "url": URL, "archive_bytes": size, "fetched_at_utc": datetime.now(UTC).isoformat(),
        "cohorts_sha256": sha256_file(ROOT / args.cohorts), "files": records})
    print(f"Fetched and verified {len(records)} files", flush=True)


if __name__ == "__main__":
    main()
