"""Statistical protocol.

This is what separates a top-tier paper from a mid one:
  - 3 seeds minimum on headline numbers, reported as mean +/- std
  - McNemar for paired classifier comparisons (as in ARC-Net, p = 0.028)
  - Bootstrap 95% CIs for IoU, ESI, AUC
  - Holm-Bonferroni across the ablation family
"""
from __future__ import annotations

from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
from scipy import stats


def mcnemar(correct_a: Sequence[bool], correct_b: Sequence[bool],
            exact_threshold: int = 25) -> Dict[str, float]:
    """Paired comparison of two classifiers on the same samples.

    Uses the exact binomial test when the discordant count is small, and the
    chi-square approximation with continuity correction otherwise.
    """
    a = np.asarray(correct_a).astype(bool)
    b = np.asarray(correct_b).astype(bool)
    if a.shape != b.shape:
        raise ValueError("inputs must be the same length")

    n01 = int(np.sum(a & ~b))   # A right, B wrong
    n10 = int(np.sum(~a & b))   # A wrong, B right
    n = n01 + n10

    if n == 0:
        return {"n01": 0, "n10": 0, "statistic": 0.0, "p_value": 1.0, "test": "none"}

    if n < exact_threshold:
        p = float(stats.binomtest(n01, n, 0.5).pvalue)
        return {"n01": n01, "n10": n10, "statistic": float(min(n01, n10)),
                "p_value": p, "test": "exact"}

    chi2 = (abs(n01 - n10) - 1) ** 2 / n
    p = float(stats.chi2.sf(chi2, df=1))
    return {"n01": n01, "n10": n10, "statistic": float(chi2),
            "p_value": p, "test": "chi2_cc"}


def bootstrap_ci(values: Sequence[float], statistic: Callable = np.mean,
                 n_resamples: int = 10000, alpha: float = 0.05,
                 seed: int = 0) -> Dict[str, float]:
    """Percentile bootstrap CI. 10,000 resamples, as specified in the plan."""
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"estimate": float("nan"), "lo": float("nan"), "hi": float("nan")}

    idx = rng.integers(0, arr.size, size=(n_resamples, arr.size))
    dist = np.array([statistic(arr[i]) for i in idx])
    return {
        "estimate": float(statistic(arr)),
        "lo": float(np.percentile(dist, 100 * alpha / 2)),
        "hi": float(np.percentile(dist, 100 * (1 - alpha / 2))),
        "n_resamples": n_resamples,
    }


def holm_bonferroni(p_values: Sequence[float], alpha: float = 0.05
                    ) -> List[Tuple[int, float, float, bool]]:
    """Holm-Bonferroni step-down correction across a family of tests.

    Returns (original_index, p_value, adjusted_p, reject) sorted by p.
    Apply across the ablation family — TIFS and CVPR reviewers do check.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = len(p)
    order = np.argsort(p)

    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        value = (m - rank) * p[i]
        running = max(running, min(value, 1.0))
        adjusted[i] = running

    return [(int(i), float(p[i]), float(adjusted[i]), bool(adjusted[i] <= alpha))
            for i in order]


def mean_std(values: Sequence[float]) -> str:
    """Format seeds as 'mean ± std' for a paper table."""
    arr = np.asarray(values, dtype=np.float64)
    return f"{arr.mean():.3f} ± {arr.std(ddof=1 if arr.size > 1 else 0):.3f}"
