import numpy as np
import pytest

from music_genre.evaluation.noise_diagnostic import representation_drift


def test_drift_distinguishes_raw_db_changes_from_clipped_inputs_and_scaled_features():
    original = np.tile([-100., 40.], (3, 1, 1))
    changed = np.tile([-80., 20.], (3, 1, 1))
    result = representation_drift(original, changed, [3., 4.], [6., 8.], [3., 2.], -80., 20.)
    assert result['channels'][0]['raw_mean_absolute_db_change'] == 20.
    assert result['channels'][0]['bounded_mean_absolute_change'] == 0.
    assert result['channels'][0]['reference_below_min_fraction'] == .5
    assert result['channels'][0]['reference_above_max_fraction'] == .5
    assert result['channels'][0]['changed_below_min_fraction'] == 0.
    assert result['embedding_l2'] == 5.
    assert result['embedding_relative_l2'] == 1.
    assert result['scaled_delta_l2'] == pytest.approx(np.sqrt(5))
    assert result['cosine_distance'] == pytest.approx(0.)


def test_zero_reference_norm_is_explicit_not_nan():
    m = np.zeros((3, 1, 2))
    result = representation_drift(m, m, [0., 0.], [1., 0.], [1., 1.], -80., 20.)
    assert result['embedding_relative_l2'] is None
    assert result['cosine_distance'] is None
    assert result['scaled_delta_l2'] == 1.


@pytest.mark.parametrize('change', ['shape', 'scale', 'nonfinite', 'bounds'])
def test_invalid_representations_are_rejected(change):
    a = np.zeros((3, 1, 2))
    b, scale, upper = a.copy(), [1., 1.], 20.
    if change == 'shape':
        b = np.zeros((3, 2, 2))
    elif change == 'scale':
        scale = [0., 1.]
    elif change == 'nonfinite':
        b[0, 0, 0] = np.nan
    else:
        upper = -80.
    with pytest.raises(ValueError):
        representation_drift(a, b, [1., 0.], [1., 0.], scale, -80., upper)
