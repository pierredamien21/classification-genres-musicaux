import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cv_v2", ROOT / "scripts/cv_v2.py")
cv_v2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cv_v2)

MLP = {"head": "mlp", "hidden_units": 8, "batch_size": 32, "learning_rate": 0.01, "max_iter": 2000,
       "tolerance": 0.0001, "n_iter_no_change": 10, "seed": 42}


def toy():
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(3), 40)
    x = rng.normal(size=(120, 5)) + 3 * y[:, None]
    fit = np.arange(120) % 4 != 0  # every class on both sides
    return x[fit], y[fit], x[~fit], y[~fit]


def test_mlp_head_returns_reproducible_probabilities_in_label_order():
    x, y, x_eval, y_eval = toy()
    fit_predict = cv_v2.make_fit(MLP, ["a", "b", "c"])
    first, second = (fit_predict(x, y, x_eval, 1.0) for _ in range(2))
    assert first.shape == (30, 3) and np.allclose(first.sum(axis=1), 1)
    assert np.array_equal(first, second)
    assert (first.argmax(axis=1) == y_eval).mean() > 0.9  # separable toy data


def test_mlp_alpha_is_the_regularization_knob():
    x, y, x_eval, _ = toy()
    fit_predict = cv_v2.make_fit(MLP, ["a", "b", "c"])
    weak, strong = (fit_predict(x, y, x_eval, alpha) for alpha in (0.01, 1000.0))
    assert strong.max() < weak.max()  # heavy weight decay flattens the probabilities
