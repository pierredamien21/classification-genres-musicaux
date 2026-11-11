"""V2 final evaluation statistics (configs/protocol_v2.json): BCa intervals, Holm, out-of-domain.

The V2 test has one track per artist, so resampling tracks resamples artists. The BCa p-value
inverts the interval: p < alpha exactly when the 1 - alpha interval excludes 0.
"""

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score


def fast_macro_f1(truth, predicted, classes):
    """Macro-F1 over a fixed class list; a class never true nor predicted scores 0 (zero_division=0)."""
    confusion = np.bincount(truth * classes + predicted, minlength=classes * classes).reshape(classes, classes)
    tp = np.diag(confusion)
    denominator = confusion.sum(axis=0) + confusion.sum(axis=1)
    return float(np.mean(np.divide(2 * tp, denominator, out=np.zeros(classes), where=denominator > 0)))


def bca(statistic, n, *, replicates, seed, level=0.95):
    """BCa bootstrap of statistic(indices) over n units, with the two-sided p-value of "statistic = 0"."""
    rng = np.random.Generator(np.random.PCG64(seed))
    point = float(statistic(np.arange(n)))
    boot = np.array([statistic(rng.integers(0, n, n)) for _ in range(replicates)])
    jack = np.array([statistic(np.delete(np.arange(n), i)) for i in range(n)])
    if not (np.isfinite(boot).all() and np.isfinite(jack).all()):
        raise ValueError("Non-finite bootstrap statistic")
    z0 = norm.ppf((np.sum(boot < point) + 0.5 * np.sum(boot == point)) / replicates)
    d = jack.mean() - jack
    a = float(np.sum(d ** 3) / (6 * np.sum(d ** 2) ** 1.5)) if np.any(d) else 0.0

    def level_of(z):
        return norm.cdf(z0 + (z0 + z) / (1 - a * (z0 + z)))

    tail = (1 - level) / 2
    low, high = np.quantile(boot, [level_of(norm.ppf(tail)), level_of(norm.ppf(1 - tail))])
    share = (np.sum(boot < 0) + 0.5 * np.sum(boot == 0)) / replicates
    w = norm.ppf(np.clip(share, 0.5 / replicates, 1 - 0.5 / replicates)) - z0
    z = w / (1 + a * w) - z0  # nominal z whose BCa bound sits at 0
    return {"point": point, "low": float(low), "high": float(high), "level": level,
            "p_value": float(2 * min(norm.cdf(z), norm.sf(z))), "bias_z0": float(z0), "acceleration": a,
            "replicates": replicates, "seed": seed}


def holm(p_values, alpha=0.05):
    """Holm step-down adjusted p-values; a hypothesis is rejected when its adjusted p < alpha."""
    order, running, result = sorted(p_values, key=p_values.get), 0.0, {}
    for rank, name in enumerate(order):
        running = min(1.0, max(running, (len(order) - rank) * p_values[name]))
        result[name] = {"p_adjusted": running, "rejected": running < alpha}
    return result


def ood_summary(in_domain, out_of_domain, threshold):
    """AUROC of a confidence score (in-domain positive) and acceptance rates at a fixed threshold."""
    scores = np.concatenate([in_domain, out_of_domain])
    truth = np.r_[np.ones(len(in_domain)), np.zeros(len(out_of_domain))]
    return {"auroc": float(roc_auc_score(truth, scores)),
            "fpr_at_threshold": float(np.mean(out_of_domain >= threshold)),
            "in_domain_retention": float(np.mean(in_domain >= threshold)),
            "in_domain": len(in_domain), "out_of_domain": len(out_of_domain)}
