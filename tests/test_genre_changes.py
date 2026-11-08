from music_genre.evaluation.errors import genre_changes


def test_genre_changes_separates_recovered_tracks_from_false_positives():
    # A loses one correct track but recovers another; its recall is unchanged.
    # A also loses a false positive from C, so its precision improves.
    result = genre_changes([0, 0, 1, 1, 2], [0, 1, 1, 2, 0], [1, 0, 1, 1, 2],
                           ["A", "B", "C"])
    assert result["A"] == {
        "tracks": 2, "both_correct": 0, "corrected": 1, "regressed": 1,
        "both_wrong": 0, "disagreements": 2,
        "reference_false_positives": 1, "candidate_false_positives": 0,
    }
    assert result["B"]["corrected"] == 1
    assert result["B"]["regressed"] == 0
    assert result["B"]["candidate_false_positives"] == 1
    assert result["C"]["corrected"] == 1
    assert result["C"]["candidate_false_positives"] == 0
