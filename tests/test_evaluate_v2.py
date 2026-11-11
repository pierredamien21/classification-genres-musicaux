import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import evaluate_v2

from music_genre.data.audit import sha256_file


def write_lock(root, **overrides):
    (root / "pinned.txt").write_text("frozen")
    (root / "reports/v2").mkdir(parents=True)
    lock = {"status": "locked", "test_evaluated": False, "environment": evaluate_v2.environment(),
            "files_sha256": {"pinned.txt": sha256_file(root / "pinned.txt")}, **overrides}
    (root / evaluate_v2.LOCK).write_text(json.dumps(lock))


def test_verify_lock_accepts_an_unchanged_tree(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_v2, "ROOT", tmp_path)
    write_lock(tmp_path)
    assert evaluate_v2.verify_lock() == sha256_file(tmp_path / evaluate_v2.LOCK)


def test_verify_lock_refuses_a_changed_file(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_v2, "ROOT", tmp_path)
    write_lock(tmp_path)
    (tmp_path / "pinned.txt").write_text("edited")
    with pytest.raises(ValueError, match="pinned files changed"):
        evaluate_v2.verify_lock()


def test_verify_lock_refuses_another_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_v2, "ROOT", tmp_path)
    write_lock(tmp_path, environment={"python": "0.0"})
    with pytest.raises(ValueError, match="environment changed"):
        evaluate_v2.verify_lock()


def test_verify_lock_refuses_a_spent_lock(tmp_path, monkeypatch):
    monkeypatch.setattr(evaluate_v2, "ROOT", tmp_path)
    write_lock(tmp_path, test_evaluated=True)
    with pytest.raises(ValueError, match="not in force"):
        evaluate_v2.verify_lock()
