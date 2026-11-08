"""The reading notebooks must not display a silently changed report."""

import hashlib

import pytest

from music_genre.presentation.notebook_data import read_verified_json


def test_reading_a_report_refuses_changed_contents(tmp_path):
    report = tmp_path / "report.json"
    original = b'{"macro_f1": 0.5}'
    report.write_bytes(original)
    expected = hashlib.sha256(original).hexdigest()
    assert read_verified_json(tmp_path, "report.json", expected) == {"macro_f1": 0.5}
    report.write_text('{"macro_f1": 0.9}', encoding="utf-8")
    with pytest.raises(ValueError, match="report.json"):
        read_verified_json(tmp_path, "report.json", expected)
