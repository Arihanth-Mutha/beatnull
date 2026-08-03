import math

import numpy as np
import pytest

from beatnull import stats


def test_exact_permutation_known_value():
    # Only 1 of the C(6,3) = 20 relabellings puts all three largest values in "a".
    r = stats.mean_difference_test([3, 4, 5], [0, 1, 2], alternative="greater")
    assert r.exact
    assert r.n_permutations == 20
    assert r.p_value == pytest.approx(1 / 20)
    assert r.min_p == pytest.approx(1 / 20)
    assert r.observed == pytest.approx(3.0)


def test_two_sided_doubles_symmetric_tail():
    r = stats.mean_difference_test([3, 4, 5], [0, 1, 2], alternative="two-sided")
    assert r.p_value == pytest.approx(2 / 20)


def test_less_is_mirror_of_greater():
    g = stats.mean_difference_test([0, 1, 2], [3, 4, 5], alternative="less")
    assert g.p_value == pytest.approx(1 / 20)


def test_sampled_test_uses_plus_one_correction():
    rng = np.random.default_rng(1)
    a, b = rng.normal(5, 1, 20), rng.normal(0, 1, 20)
    r = stats.mean_difference_test(a, b, n_resamples=999, rng=rng)
    assert not r.exact
    assert r.p_value == pytest.approx(1 / 1000)


def test_stratified_labels_never_cross_strata():
    # Two strata with very different levels. Pooled shuffling would mix them and make
    # the within-stratum effect look noisier; stratified shuffling keeps it sharp.
    a = np.array([10.5, 10.6, 10.7, 0.5, 0.6, 0.7])
    b = np.array([10.0, 10.1, 10.2, 0.0, 0.1, 0.2])
    sa = np.array(["x", "x", "x", "y", "y", "y"])
    strat = stats.mean_difference_test(a, b, strata_a=sa, strata_b=sa.copy(),
                                       n_resamples=5000, rng=np.random.default_rng(0))
    pooled = stats.mean_difference_test(a, b, n_resamples=5000, rng=np.random.default_rng(0))
    assert strat.p_value < pooled.p_value
    assert strat.p_value <= 1 / 400 + 1e-3  # exact minimum is 1/(20*20)


def test_nan_is_rejected_not_zeroed():
    with pytest.raises(ValueError, match="NaN"):
        stats.mean_difference_test([1.0, math.nan], [0.0, 0.1])


def test_holm_matches_reference():
    # Reference values computed by hand: sorted p = .01, .02, .04 with m = 3.
    assert stats.holm([0.04, 0.01, 0.02]) == pytest.approx([0.04, 0.03, 0.04])
    assert stats.holm([0.5, 0.9]) == pytest.approx([1.0, 1.0])


def test_hedges_g_sign_and_small_sample_shrinkage():
    a, b = np.array([2.0, 3.0, 4.0]), np.array([0.0, 1.0, 2.0])
    g = stats.hedges_g(a, b)
    assert g > 0
    assert g < 2.0  # Cohen's d is exactly 2 here; the correction shrinks it


def test_bootstrap_interval_contains_true_difference():
    rng = np.random.default_rng(3)
    a, b = rng.normal(1, 1, 200), rng.normal(0, 1, 200)
    ci = stats.bootstrap_mean_difference(a, b, rng=rng)
    assert ci.low < 1.0 < ci.high


def test_trend_test_detects_monotone_drift():
    x = np.arange(30)
    y = x + np.random.default_rng(0).normal(0, 3, 30)
    r = stats.trend_test(x, y, alternative="greater", n_resamples=2000)
    assert r.observed > 0.7
    assert r.p_value < 0.01


def test_slope_difference_ignores_level_difference():
    x = np.arange(20, dtype=float)
    noise = np.random.default_rng(1).normal(0, 0.5, 20)
    # Same slope, very different level: the test must not call this a slope difference.
    r = stats.slope_difference_test(x, 0.1 * x + 100 + noise, x, 0.1 * x + noise[::-1],
                                    n_resamples=2000)
    assert r.p_value > 0.05


def test_single_look_is_nominal():
    d = stats.sequential_design(1, 0.05, "two-sided")
    assert d.naive_false_positive_rate == 0.05
    assert d.per_look_alpha == 0.05


def test_pocock_boundary_matches_published_value():
    # Pocock (1977): 5 equally spaced two-sided looks at overall alpha 0.05 need a
    # nominal per-look alpha of 0.0158; testing each at 0.05 gives about 14%.
    d = stats.sequential_design(5, 0.05, "two-sided")
    assert d.per_look_alpha == pytest.approx(0.0158, abs=0.0008)
    assert d.naive_false_positive_rate == pytest.approx(0.142, abs=0.006)


@pytest.mark.slow
def test_false_positive_rate_is_calibrated_under_the_null():
    rng = np.random.default_rng(11)
    trials, alpha, rejections = 600, 0.05, 0
    for _ in range(trials):
        a, b = rng.normal(0, 1, 8), rng.normal(0, 1, 8)
        r = stats.mean_difference_test(a, b, n_resamples=999, rng=rng)
        rejections += r.p_value < alpha
    rate = rejections / trials
    # 99.9% binomial band around 0.05 for 600 trials is roughly [0.025, 0.08].
    assert 0.025 <= rate <= 0.08
