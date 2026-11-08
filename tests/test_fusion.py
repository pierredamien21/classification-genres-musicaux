from copy import deepcopy

import numpy as np
import pytest

from music_genre.features.fusion import concatenate_features


def inputs():
    index = [
        {"track_id": "1", "window_id": "0", "row_index": "0", "split": "train"},
        {"track_id": "2", "window_id": "0", "row_index": "1", "split": "validation"},
    ]
    return (np.array([[1., 2.], [100., 200.]]), np.array([[3., 4., 5.], [300., 400., 500.]]),
            index, deepcopy(index), ["v0", "v1"], ["t", "h", "r"], ["timbre", "harmony", "rhythm"])


@pytest.mark.parametrize("excluded,columns", [(None, [0, 1, 2]), ("rhythm", [0, 1]), ("harmony", [0, 2])])
def test_fusion_preserves_raw_values_and_family_order(excluded, columns):
    args = inputs()
    matrix, names, families, selected = concatenate_features(*args, excluded)
    assert selected == columns
    np.testing.assert_array_equal(matrix[:, :2], args[0])
    np.testing.assert_array_equal(matrix[:, 2:], args[1][:, columns])
    assert families == ["vit", "vit"] + [args[6][i] for i in columns]
    assert names == ["v0", "v1"] + [f"descriptor/{args[5][i]}" for i in columns]


def test_fusion_rejects_silent_row_misalignment():
    args = list(inputs())
    args[3] = list(reversed(args[3]))
    with pytest.raises(ValueError, match="identical indexed windows"):
        concatenate_features(*args)


def test_fusion_rejects_test_even_when_both_indexes_match():
    args = list(inputs())
    args[2][1]["split"] = "test"
    args[3] = deepcopy(args[2])
    with pytest.raises(ValueError, match="Only training and validation"):
        concatenate_features(*args)


def test_fusion_rejects_duplicate_windows():
    args = list(inputs())
    args[2][1]["track_id"] = "1"
    args[3] = deepcopy(args[2])
    with pytest.raises(ValueError, match="Duplicate"):
        concatenate_features(*args)


def test_fusion_rejects_nonfinite_features():
    args = list(inputs())
    args[1][1, 0] = np.nan
    with pytest.raises(ValueError, match="Invalid feature matrices"):
        concatenate_features(*args)


def test_fusion_rejects_unknown_ablation():
    with pytest.raises(ValueError, match="Unsupported excluded family"):
        concatenate_features(*inputs(), "other")
