import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("audit_v2_cohorts", ROOT / "scripts/audit_v2_cohorts.py")
audit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(audit)


def row(track, role, rank, genre="Folk"):
    return {"track_id": track, "role": role, "rank": str(rank), "genre": genre, "artist_id": track}


def test_failed_primary_takes_next_passing_spare_in_rank_order():
    rows = [row("a", "test", 0), row("b", "test", 1), row("c", "test_spare", 2),
            row("d", "test_spare", 3), row("e", "test_spare", 4)]
    passed = {"a": True, "b": False, "c": False, "d": True, "e": True}
    final = audit.replace(rows, passed)
    assert [(r["track_id"], r["role"], r["replacement"]) for r in final] == [
        ("a", "test", False), ("d", "test", True)]


def test_genre_without_enough_spares_shrinks_and_genres_stay_separate():
    rows = [row("a", "ood", 0, "Blues"), row("b", "ood", 1, "Blues"),
            row("c", "ood", 0, "Jazz"), row("d", "ood_spare", 1, "Jazz")]
    passed = {"a": False, "b": True, "c": False, "d": True}
    final = audit.replace(rows, passed)
    assert sorted(r["track_id"] for r in final) == ["b", "d"]
