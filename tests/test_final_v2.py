import numpy as np
import pytest
from scipy import stats
from sklearn.metrics import f1_score

from music_genre.evaluation.final_v2 import bca, fast_macro_f1, holm, ood_summary


def test_fast_macro_f1_matches_scikit_learn_with_a_missing_class():
    rng = np.random.default_rng(0)
    truth, predicted = rng.integers(0, 7, 300), rng.integers(0, 8, 300)  # class 7 never true
    assert fast_macro_f1(truth, predicted, 8) == pytest.approx(
        f1_score(truth, predicted, labels=range(8), average="macro", zero_division=0))


def test_bca_interval_agrees_with_scipy_on_a_skewed_mean():
    data = np.random.default_rng(1).exponential(size=200)
    ours = bca(lambda i: data[i].mean(), len(data), replicates=5000, seed=3)
    theirs = stats.bootstrap((data,), np.mean, n_resamples=5000, method="BCa",
                             rng=np.random.default_rng(4)).confidence_interval
    width = theirs.high - theirs.low
    assert ours["point"] == pytest.approx(data.mean())
    assert ours["low"] == pytest.approx(theirs.low, abs=0.05 * width)
    assert ours["high"] == pytest.approx(theirs.high, abs=0.05 * width)


@pytest.mark.parametrize("shift", [0.0, 0.08, 0.12, 0.2, 0.4])
def test_bca_p_value_is_below_alpha_exactly_when_the_interval_excludes_zero(shift):
    data = np.random.default_rng(2).normal(shift, 1, size=400)
    result = bca(lambda i: data[i].mean(), len(data), replicates=4000, seed=5)
    excludes_zero = result["low"] > 0 or result["high"] < 0
    assert excludes_zero == (result["p_value"] < 0.05)


def test_holm_on_worked_examples():
    both = holm({"H2": 0.01, "H3": 0.04}, alpha=0.05)
    assert both["H2"] == {"p_adjusted": pytest.approx(0.02), "rejected": True}
    assert both["H3"] == {"p_adjusted": pytest.approx(0.04), "rejected": True}
    neither = holm({"H2": 0.04, "H3": 0.03}, alpha=0.05)
    assert neither["H3"] == {"p_adjusted": pytest.approx(0.06), "rejected": False}
    assert neither["H2"] == {"p_adjusted": pytest.approx(0.06), "rejected": False}


def test_ood_summary_on_a_worked_example():
    result = ood_summary(np.array([0.9, 0.8, 0.7]), np.array([0.75, 0.2]), threshold=0.5)
    assert result == {"auroc": pytest.approx(5 / 6), "fpr_at_threshold": 0.5,
                      "in_domain_retention": 1.0, "in_domain": 3, "out_of_domain": 2}
