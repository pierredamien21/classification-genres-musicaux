import pytest

from music_genre.data.review import review_rows

SETTINGS = {"minimum_words": 3, "minimum_unique_words": 3}


def make_row(key, genre="Pop", split="train", pcm="same", flags="duplicate_pcm_sha256"):
    return {"track_id": key, "genre": genre, "split": split, "pcm_sha256": pcm,
            "flags": flags, "status": "needs_review", "decode_status": "ok",
            "exclusion_reason": ""}


def run_review(rows, pairs=()):
    fp = {r["track_id"]: {"fingerprint": [1, 2, 3], "returncode": 0, "diagnostic": ""}
          for r in rows}
    return review_rows(rows, fp, pairs, SETTINGS)


def test_exact_duplicate_lowest_eligible_retained():
    result = run_review([make_row("003"), make_row("001"), make_row("002")])
    assert [r["decision"] for r in result] == ["keep", "exclude", "exclude"]
    assert all(r["representative_track_id"] == "001" for r in result)


def test_conflicting_labels_and_cross_split_quarantined():
    assert all(r["decision"] == "hold" for r in
               run_review([make_row("001"), make_row("002", genre="Rock")]))
    assert all(r["decision"] == "hold" for r in
               run_review([make_row("001"), make_row("002", split="test")]))


def test_near_candidate_holds_whole_connected_component():
    rows = [make_row("001"), make_row("002"), make_row("003", pcm="different", flags="")]
    result = run_review(rows, [{"a": "002", "b": "003", "exact_pcm": False}])
    assert all(r["decision"] == "hold" for r in result)


def test_level_flag_retained_and_unknown_flags_held():
    rows = [make_row("001", pcm="a", flags="near_full_scale"),
            make_row("002", pcm="b", flags="unrecognized_future_flag")]
    result = run_review(rows)
    assert [r["decision"] for r in result] == ["keep", "hold"]


def test_fingerprint_success_does_not_override_reference_decode_failure():
    row = make_row("001", pcm="", flags="decoder_diagnostic")
    row.update(status="excluded", decode_status="failed", exclusion_reason="decode_failed")
    result = run_review([row])[0]
    assert result["decision"] == "exclude"
    assert "decode_failed" in result["decision_reasons"]


def test_inconsistent_track_lists_rejected():
    with pytest.raises(ValueError, match="IDs"):
        review_rows([make_row("001")], {}, [], SETTINGS)
