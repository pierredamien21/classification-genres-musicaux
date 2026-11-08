"""Acquire the official FMA-small archives, verify and safely extract them.

Standard-library only; curl supplies resumable HTTPS downloads on Windows.
Run from any directory with a working Python interpreter.
"""

import hashlib
import json
import shutil
import stat
import subprocess
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
ARCHIVES = RAW / "archives"
REPORT = ROOT / "data" / "manifests" / "acquisition_fma.json"
SOURCE = "https://github.com/mdeff/fma"
FILES = (
    ("fma_metadata", "f0df49ffe5f2a6008d7dc83c6915b31835dfe733"),
    ("fma_small", "ade154f733639d52e35e32f5593efe5be76c6d70"),
)


def log(message):
    print(message, flush=True)


def hashes(path):
    sha1 = hashlib.sha1()
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            sha1.update(block)
            sha256.update(block)
    return sha1.hexdigest(), sha256.hexdigest()


def download(url, path):
    curl = shutil.which("curl.exe") or shutil.which("curl")
    if not curl:
        raise RuntimeError("curl is required for resumable downloads")
    process = subprocess.Popen([
        curl, "--location", "--fail", "--silent", "--show-error",
        "--proto", "=https", "--proto-redir", "=https",
        "--connect-timeout", "30", "--retry", "5", "--retry-delay", "5",
        "--speed-limit", "1024", "--speed-time", "60",
        "--continue-at", "-", "--output", str(path), url,
    ])
    try:
        while True:
            try:
                code = process.wait(timeout=15)
                break
            except subprocess.TimeoutExpired:
                size = path.stat().st_size if path.exists() else 0
                log(f"Downloading {path.name}: {size / 2**20:.1f} MiB")
        if code:
            raise RuntimeError(f"curl failed ({code}); partial file retained: {path}")
    finally:
        if process.poll() is None:
            process.terminate()
            process.wait()


def extract(path, expected_folder):
    file_count = 0
    mp3_count = 0
    with zipfile.ZipFile(path) as archive:
        members = archive.infolist()
        # Validate all paths before writing; never trust archive paths.
        for member in members:
            relative = PurePosixPath(member.filename)
            target = (RAW / member.filename).resolve()
            if (relative.is_absolute() or ".." in relative.parts
                    or "\\" in member.filename or ":" in member.filename
                    or not target.is_relative_to(RAW.resolve())
                    or stat.S_ISLNK(member.external_attr >> 16)):
                raise RuntimeError(f"Unsafe archive member: {member.filename}")
            if relative.parts and relative.parts[0] not in (expected_folder, "__MACOSX"):
                raise RuntimeError(f"Unexpected archive member: {member.filename}")
        required = sum(m.file_size for m in members)
        if shutil.disk_usage(RAW).free < required + 2**30:
            raise RuntimeError("Insufficient free disk space for extraction")
        for member in members:
            relative = PurePosixPath(member.filename)
            if not relative.parts or relative.parts[0] == "__MACOSX":
                continue
            target = RAW / member.filename
            if member.is_dir():
                target.mkdir(parents=True, exist_ok=True)
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            # Reading each member fully also checks its ZIP CRC.
            with archive.open(member) as source, target.open("wb") as destination:
                shutil.copyfileobj(source, destination, length=1024 * 1024)
            file_count += 1
            mp3_count += target.suffix.lower() == ".mp3"
            if file_count % 500 == 0:
                log(f"Extracting {expected_folder}: {file_count} files")
    return file_count, mp3_count


def main():
    ARCHIVES.mkdir(parents=True, exist_ok=True)
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "source": SOURCE,
        "started_at_utc": datetime.now(UTC).isoformat(),
        "status": "running",
        "archives": [],
        "audio_decoding_audit": "not_performed",
    }

    def save():
        REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    save()
    try:
        for name, expected_sha1 in FILES:
            url = f"https://os.unil.cloud.switch.ch/fma/{name}.zip"
            final = ARCHIVES / f"{name}.zip"
            partial = ARCHIVES / f"{name}.zip.part"
            if not final.exists():
                log(f"Starting {name} from {url}")
                download(url, partial)
                log(f"Verifying {name}")
                sha1, sha256 = hashes(partial)
                if sha1 != expected_sha1:
                    raise RuntimeError(f"SHA1 mismatch for {name}; partial retained")
                partial.rename(final)
            else:
                log(f"Verifying existing {name}")
                sha1, sha256 = hashes(final)
                if sha1 != expected_sha1:
                    raise RuntimeError(f"SHA1 mismatch for existing {final}")
            count, mp3_count = extract(final, name)
            if name == "fma_small" and mp3_count != 8000:
                raise RuntimeError(f"Expected 8000 MP3 files, found {mp3_count}")
            record = {
                "url": url, "archive": final.relative_to(ROOT).as_posix(),
                "bytes": final.stat().st_size, "sha1": sha1, "sha256": sha256,
                "official_sha1_verified": True, "zip_crc_verified": True,
                "extracted_to": (RAW / name).relative_to(ROOT).as_posix(),
                "files": count, "mp3_files": mp3_count,
            }
            report["archives"].append(record)
            save()
            log(f"Ready: {name}, {count} files, {mp3_count} MP3")
        report["status"] = "complete"
        report["completed_at_utc"] = datetime.now(UTC).isoformat()
        save()
        log(f"Acquisition complete. Report: {REPORT}")
    except Exception as error:
        report["status"] = "failed"
        report["error"] = str(error)
        save()
        raise


if __name__ == "__main__":
    main()
