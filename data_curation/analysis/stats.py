"""Shared rank-based statistics for the E1/E2/E3 dataset-review analyses (`rubric_results.py`,
`repair_figures.py`, `baseline_reliability_figures.py`). The rubric scores are ordinal 1-5, so
every function here works on ranks/levels rather than means, matching the methodology described
in `paper/main.tex` (Section: Why a rubric judge instead of a metric). Local, CPU-only — pure
Python plus numpy for the bootstrap resampling, no GPU/API.
"""

from __future__ import annotations

from collections import Counter

import numpy as np


def cliffs_delta(a: list[int], b: list[int]) -> float:
    """Cliff's delta of `a` vs `b`: (P(a>b) - P(a<b)), in [-1, 1].

    Computed from the 1-5 level counts rather than an O(n*m) pairwise loop, since the rubric
    only has 5 discrete levels — this stays fast even at the thousands-of-rows scale of E1.
    Negative means `a` tends to score lower than `b` (e.g. a defective stratum vs.\\ S0 clean).
    """
    if not a or not b:
        return 0.0
    count_a, count_b = Counter(a), Counter(b)
    levels = sorted(set(count_a) | set(count_b))
    greater = less = 0
    for i in levels:
        for j in levels:
            if i > j:
                greater += count_a[i] * count_b[j]
            elif i < j:
                less += count_a[i] * count_b[j]
    return (greater - less) / (len(a) * len(b))


def bootstrap_cliffs_delta_ci(
    a: list[int], b: list[int], *, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for `cliffs_delta(a, b)`, resampling both groups with replacement.

    Used for the F4 effect-size forest plot, where the whisker width is the entire point: at
    thousands of rows every E1 comparison is significant, so the CI (not a p-value) is what a
    reader should look at to judge whether an effect is real.
    """
    rng = np.random.default_rng(seed)
    a_arr, b_arr = np.asarray(a), np.asarray(b)
    deltas = np.empty(n_boot)
    for i in range(n_boot):
        a_sample = rng.choice(a_arr, size=len(a_arr), replace=True).tolist()
        b_sample = rng.choice(b_arr, size=len(b_arr), replace=True).tolist()
        deltas[i] = cliffs_delta(a_sample, b_sample)
    lo, hi = np.percentile(deltas, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def median(values: list[int]) -> float:
    """Median of an ordinal score list; thin wrapper so callers don't need `statistics` directly."""
    import statistics
    return float(statistics.median(values))


def bootstrap_median_ci(
    values: list[float], *, n_boot: int = 500, seed: int = 0, alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the median of `values`. Used for the F5 length-bias ribbons,
    where each bin's median needs an uncertainty band without assuming a parametric distribution.
    """
    if not values:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.asarray(values)
    medians = np.median(rng.choice(arr, size=(n_boot, len(arr)), replace=True), axis=1)
    lo, hi = np.percentile(medians, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


def wilson_ci(successes: int, n: int, *, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion, as a percentage pair.

    Used for the E3 head-to-head win rate (F7), where a normal-approximation interval would
    misbehave near the placebo's near-100% tie rate.
    """
    if n == 0:
        return (0.0, 0.0)
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = p + z ** 2 / (2 * n)
    spread = z * ((p * (1 - p) / n + z ** 2 / (4 * n ** 2)) ** 0.5)
    lo = (centre - spread) / denom
    hi = (centre + spread) / denom
    return (100 * max(0.0, lo), 100 * min(1.0, hi))
