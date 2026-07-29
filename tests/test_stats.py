"""Tests for data_curation.analysis.stats: Cliff's delta and its bootstrap CI on small synthetic
score lists. Pure computation, no artifacts, no network/GPU.
"""
from data_curation.analysis.stats import bootstrap_cliffs_delta_ci, cliffs_delta


def test_cliffs_delta_is_zero_for_identical_distributions():
    a = [1, 2, 3, 4, 5]
    b = [1, 2, 3, 4, 5]
    assert cliffs_delta(a, b) == 0.0


def test_cliffs_delta_is_negative_one_when_a_is_strictly_worse():
    a = [1, 1, 1]
    b = [5, 5, 5]
    assert cliffs_delta(a, b) == -1.0


def test_cliffs_delta_is_positive_one_when_a_is_strictly_better():
    a = [5, 5, 5]
    b = [1, 1, 1]
    assert cliffs_delta(a, b) == 1.0


def test_cliffs_delta_matches_known_pipe_defect_signature():
    # S2 multi-pipe vs. S0 clean on single-focus (main.tex Table 1): median 1 vs. 5, delta ~ -0.95.
    s2 = [1] * 90 + [2] * 10
    s0 = [5] * 95 + [4] * 5
    delta = cliffs_delta(s2, s0)
    assert delta < -0.9


def test_cliffs_delta_handles_empty_group_without_raising():
    assert cliffs_delta([], [1, 2, 3]) == 0.0
    assert cliffs_delta([1, 2, 3], []) == 0.0


def test_bootstrap_cliffs_delta_ci_brackets_the_point_estimate():
    a = [1, 2, 2, 3, 1, 2, 3, 2, 1, 2]
    b = [4, 5, 5, 4, 3, 5, 4, 5, 4, 5]
    point = cliffs_delta(a, b)
    lo, hi = bootstrap_cliffs_delta_ci(a, b, n_boot=500)
    assert lo <= point <= hi
    assert -1.0 <= lo <= hi <= 1.0
