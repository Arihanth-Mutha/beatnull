"""Resampling statistics with the experimental block as the unit of inference.

Every test here is a permutation or bootstrap procedure, so the only assumption is
exchangeability under the null. Nothing is assumed about normality, and the smallest
achievable p-value is always reported, because a design with three blocks per
condition cannot reach p < 0.05 no matter how strong the effect is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import numpy as np
from scipy import stats as sps

Alternative = Literal["greater", "less", "two-sided"]
ALTERNATIVES: tuple[str, ...] = ("greater", "less", "two-sided")

EXACT_LIMIT = 20_000
"""Enumerate every relabelling when there are at most this many; sample otherwise."""


@dataclass(frozen=True)
class PermutationResult:
    observed: float
    p_value: float
    null: np.ndarray
    exact: bool
    n_permutations: int
    min_p: float
    """Smallest p-value this design can produce (resolution of the test)."""


@dataclass(frozen=True)
class Interval:
    low: float
    high: float
    level: float


def _count_extreme(null: np.ndarray, observed: float, alternative: Alternative) -> int:
    # A relative tolerance stops float noise from deciding ties.
    tol = 1e-9 * max(1.0, abs(observed))
    if alternative == "greater":
        return int(np.count_nonzero(null >= observed - tol))
    if alternative == "less":
        return int(np.count_nonzero(null <= observed + tol))
    return int(np.count_nonzero(np.abs(null) >= abs(observed) - tol))


def _check_alternative(alternative: str) -> None:
    if alternative not in ALTERNATIVES:
        raise ValueError(f"alternative must be one of {ALTERNATIVES}, got {alternative!r}")


def mean_difference_test(
    a: np.ndarray,
    b: np.ndarray,
    *,
    alternative: Alternative = "greater",
    strata_a: np.ndarray | None = None,
    strata_b: np.ndarray | None = None,
    n_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> PermutationResult:
    """Permutation test of mean(a) - mean(b).

    With strata, labels are only exchanged between blocks of the same stratum (for
    example, the same suite or culture), which keeps between-culture differences out
    of the null distribution.
    """
    _check_alternative(alternative)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.size == 0 or b.size == 0:
        raise ValueError("both samples need at least one block")
    if np.isnan(a).any() or np.isnan(b).any():
        raise ValueError("samples contain NaN; drop or report missing blocks before testing")
    rng = rng if rng is not None else np.random.default_rng(0)
    observed = float(a.mean() - b.mean())
    pooled = np.concatenate([a, b])
    n_a = a.size

    if strata_a is not None or strata_b is not None:
        if strata_a is None or strata_b is None:
            raise ValueError("give strata for both samples or for neither")
        return _stratified_test(
            pooled, n_a, np.concatenate([strata_a, strata_b]), observed, alternative,
            n_resamples, rng,
        )

    total = math.comb(pooled.size, n_a)
    if total <= EXACT_LIMIT:
        idx = np.array(list(combinations(range(pooled.size), n_a)), dtype=np.intp)
        mask = np.zeros((total, pooled.size), dtype=bool)
        np.put_along_axis(mask, idx, True, axis=1)
        sum_a = (mask * pooled).sum(axis=1)
        null = sum_a / n_a - (pooled.sum() - sum_a) / (pooled.size - n_a)
        p = _count_extreme(null, observed, alternative) / total
        extreme = (
            null.max() if alternative == "greater"
            else null.min() if alternative == "less"
            else np.abs(null).max()
        )
        min_p = _count_extreme(null, float(extreme), alternative) / total
        return PermutationResult(observed, p, null, True, total, min_p)

    keys = rng.random((n_resamples, pooled.size))
    perm = np.argsort(keys, axis=1)
    shuffled = pooled[perm]
    null = shuffled[:, :n_a].mean(axis=1) - shuffled[:, n_a:].mean(axis=1)
    p = (_count_extreme(null, observed, alternative) + 1) / (n_resamples + 1)
    min_p = max(1 / (n_resamples + 1), (1 if alternative != "two-sided" else 2) / total)
    return PermutationResult(observed, p, null, False, n_resamples, min_p)


def _stratified_test(
    pooled: np.ndarray,
    n_a: int,
    strata: np.ndarray,
    observed: float,
    alternative: Alternative,
    n_resamples: int,
    rng: np.random.Generator,
) -> PermutationResult:
    is_a = np.zeros(pooled.size, dtype=bool)
    is_a[:n_a] = True
    labels = np.empty((n_resamples, pooled.size), dtype=bool)
    total = 1
    for s in np.unique(strata):
        cols = np.flatnonzero(strata == s)
        k = int(is_a[cols].sum())
        total *= math.comb(cols.size, k)
        keys = rng.random((n_resamples, cols.size))
        ranks = np.argsort(np.argsort(keys, axis=1), axis=1)
        labels[:, cols] = ranks < k
    n_b = pooled.size - n_a
    sum_a = (labels * pooled).sum(axis=1)
    null = sum_a / n_a - (pooled.sum() - sum_a) / n_b
    p = (_count_extreme(null, observed, alternative) + 1) / (n_resamples + 1)
    min_p = max(1 / (n_resamples + 1), (1 if alternative != "two-sided" else 2) / total)
    return PermutationResult(observed, p, null, False, n_resamples, min(1.0, min_p))


def bootstrap_mean_difference(
    a: np.ndarray,
    b: np.ndarray,
    *,
    level: float = 0.95,
    n_resamples: int = 5_000,
    rng: np.random.Generator | None = None,
) -> Interval:
    """Percentile interval for mean(a) - mean(b), resampling whole blocks."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    rng = rng if rng is not None else np.random.default_rng(0)
    ia = rng.integers(0, a.size, size=(n_resamples, a.size))
    ib = rng.integers(0, b.size, size=(n_resamples, b.size))
    diffs = a[ia].mean(axis=1) - b[ib].mean(axis=1)
    tail = (1 - level) / 2
    low, high = np.quantile(diffs, [tail, 1 - tail])
    return Interval(float(low), float(high), level)


def hedges_g(a: np.ndarray, b: np.ndarray) -> float:
    """Standardised mean difference with the small-sample correction."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n1, n2 = a.size, b.size
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled_var = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    if pooled_var == 0:
        return float("nan")
    correction = 1 - 3 / (4 * (n1 + n2) - 9)
    return float((a.mean() - b.mean()) / math.sqrt(pooled_var) * correction)


def holm(p_values: list[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values (strong family-wise error control)."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def slope(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x - x.mean()
    denom = float((dx * dx).sum())
    if denom == 0:
        return 0.0
    return float((dx * (y - y.mean())).sum() / denom)


def _row_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    dx = x - x.mean(axis=1, keepdims=True)
    dy = y - y.mean(axis=1, keepdims=True)
    denom = (dx * dx).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denom > 0, (dx * dy).sum(axis=1) / denom, 0.0)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = sps.rankdata(x)
    ry = sps.rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def trend_test(
    position: np.ndarray,
    value: np.ndarray,
    *,
    alternative: Alternative = "two-sided",
    n_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> PermutationResult:
    """Permutation test of the Spearman correlation between position and value."""
    _check_alternative(alternative)
    rng = rng if rng is not None else np.random.default_rng(0)
    position = np.asarray(position, dtype=float)
    value = np.asarray(value, dtype=float)
    observed = spearman(position, value)
    rp = sps.rankdata(position)
    rv = sps.rankdata(value)
    keys = rng.random((n_resamples, rv.size))
    shuffled = rv[np.argsort(keys, axis=1)]
    rp_c = rp - rp.mean()
    sv = shuffled - shuffled.mean(axis=1, keepdims=True)
    denom = np.sqrt((rp_c**2).sum() * (sv**2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        null = np.where(denom > 0, (sv * rp_c).sum(axis=1) / denom, 0.0)
    p = (_count_extreme(null, observed, alternative) + 1) / (n_resamples + 1)
    total = math.factorial(min(rv.size, 170))
    min_p = max(1 / (n_resamples + 1), 1 / total)
    return PermutationResult(observed, p, null, False, n_resamples, min_p)


def slope_difference_test(
    x_a: np.ndarray,
    y_a: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    *,
    alternative: Alternative = "greater",
    n_resamples: int = 10_000,
    rng: np.random.Generator | None = None,
) -> PermutationResult:
    """Is the trend in sample a steeper than in sample b?

    Each sample is centred first so a difference in level cannot masquerade as a
    difference in slope; condition labels are then exchanged between blocks.
    """
    _check_alternative(alternative)
    rng = rng if rng is not None else np.random.default_rng(0)
    x = np.concatenate([x_a, x_b]).astype(float)
    y = np.concatenate([y_a - np.mean(y_a), y_b - np.mean(y_b)]).astype(float)
    n_a = len(x_a)
    observed = slope(x_a, y_a) - slope(x_b, y_b)
    perm = np.argsort(rng.random((n_resamples, x.size)), axis=1)
    xs, ys = x[perm], y[perm]
    null = _row_slopes(xs[:, :n_a], ys[:, :n_a]) - _row_slopes(xs[:, n_a:], ys[:, n_a:])
    p = (_count_extreme(null, observed, alternative) + 1) / (n_resamples + 1)
    total = math.comb(x.size, n_a)
    min_p = max(1 / (n_resamples + 1), 1 / total)
    return PermutationResult(observed, p, null, False, n_resamples, min_p)


@dataclass(frozen=True)
class SequentialDesign:
    looks: int
    alpha: float
    alternative: Alternative
    naive_false_positive_rate: float
    """Chance of a false positive if every look is tested at the nominal alpha."""
    per_look_alpha: float
    """Constant (Pocock-type) per-look threshold that keeps the overall rate at alpha."""


def sequential_design(
    looks: int,
    alpha: float = 0.05,
    alternative: Alternative = "greater",
    *,
    n_paths: int = 200_000,
    seed: int = 0,
) -> SequentialDesign:
    """Simulate equally spaced interim looks at accumulating data under the null.

    Peeking at the data and stopping when p < alpha inflates the false positive rate;
    this measures by how much, and finds the constant per-look threshold (a Pocock
    boundary) that restores it.
    """
    _check_alternative(alternative)
    if looks < 1:
        raise ValueError("looks must be >= 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    if looks == 1:
        return SequentialDesign(1, alpha, alternative, alpha, alpha)
    rng = np.random.default_rng(seed)
    increments = rng.standard_normal((n_paths, looks))
    z = np.cumsum(increments, axis=1) / np.sqrt(np.arange(1, looks + 1))
    stat = np.abs(z) if alternative == "two-sided" else (z if alternative == "greater" else -z)
    peak = stat.max(axis=1)
    tail = alpha / 2 if alternative == "two-sided" else alpha
    nominal_crit = float(sps.norm.isf(tail))
    naive = float(np.mean(peak > nominal_crit))
    crit = float(np.quantile(peak, 1 - alpha))
    per_look = float(sps.norm.sf(crit)) * (2 if alternative == "two-sided" else 1)
    return SequentialDesign(looks, alpha, alternative, naive, per_look)
