import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fetch_v2_audio", ROOT / "scripts/fetch_v2_audio.py")
fetcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fetcher)


def archive_bytes():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("fma_large/000/000002.mp3", b"audio-a" * 1000)
        archive.writestr("fma_large/155/155066.mp3", b"audio-b")
    return buffer.getvalue()


def test_range_reads_extract_only_selected_members(tmp_path):
    data, calls = archive_bytes(), []

    def fetch(start, stop):
        calls.append((start, stop))
        return data[start:stop]

    with zipfile.ZipFile(fetcher.RangeFile(fetch, len(data))) as archive:
        records = fetcher.fetch_members(archive, ["fma_large/155/155066.mp3"], tmp_path)
    assert (tmp_path / "155/155066.mp3").read_bytes() == b"audio-b"
    assert not (tmp_path / "000").exists()
    assert records[0]["bytes"] == 7 and len(records[0]["sha256"]) == 64
    assert all(0 <= start < stop <= len(data) for start, stop in calls)


def test_corrupted_member_is_rejected(tmp_path):
    data = bytearray(archive_bytes())
    position = data.index(b"audio-b")
    data[position] ^= 0xFF  # flip one stored byte; the CRC-32 no longer matches
    with (zipfile.ZipFile(fetcher.RangeFile(lambda a, b: bytes(data[a:b]), len(data))) as archive,
          pytest.raises(zipfile.BadZipFile)):
        fetcher.fetch_members(archive, ["fma_large/155/155066.mp3"], tmp_path)
    assert not (tmp_path / "155/155066.mp3").exists()
