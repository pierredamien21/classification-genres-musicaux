from copy import deepcopy

import pytest

from music_genre.features.windows import central_window_rows


def rows():
    return [
        {"row_index": str(3 * track + window), "track_id": str(track), "window_id": str(window),
         "start_seconds": str(10.0 * window), "valid_seconds": "10.0", "padded_seconds": "0.0",
         "split": "train" if track == 0 else "validation", "label_index": str(track),
         "artist_id": str(track), "source_sha256": str(track)}
        for track in range(2) for window in range(3)
    ]


def test_central_subset_keeps_each_track_and_metadata_without_mutation():
    index = rows()
    original = deepcopy(index)
    positions, result = central_window_rows(index)
    assert positions == [1, 4]
    assert result == [{**index[1], "row_index": "0"}, {**index[4], "row_index": "1"}]
    assert index == original


@pytest.mark.parametrize("kind", ["missing", "duplicate_start", "duplicate_id", "test", "label", "row_order"])
def test_central_selection_refuses_invalid_parent(kind):
    index = rows()
    if kind == "missing":
        index.pop()
    elif kind == "duplicate_start":
        index[0]["start_seconds"] = "10.0"
    elif kind == "duplicate_id":
        index[0]["window_id"] = "1"
    elif kind == "test":
        index[0]["split"] = "test"
    elif kind == "label":
        index[0]["label_index"] = "7"
    else:
        index[0]["row_index"] = "2"
    with pytest.raises(ValueError):
        central_window_rows(index)
