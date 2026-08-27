"""The checks behind a verdict.

Each check answers one way a closed-loop result can fool you, and returns a
``CheckResult`` with a status, a one-line headline and a plain explanation:

``integrity``      missing data that differs between conditions
``controls``       does real beat every control, blocks as the unit, Holm-corrected
``replication``    what a trial-level (pseudoreplicated) test would have claimed
``order``          drift across the run, and whether the schedule confounds it
``improvement``    is improvement over time specific to the real condition
``sequential``     false positives from looking at the data more than once
``plan``           was the analysis pre-registered, and is the plan intact
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy import stats as sps

from . import stats
from .data import DataError, Dataset
from .plan import PlanStatus

PASS, FAIL, WARN, INFO, INCONCLUSIVE, SKIPPED = (
    "pass", "fail", "warn", "info", "inconclusive", "skipped",
)

IMBALANCE_LIMIT = 0.10
"""A condition whose mean schedule position is off-centre by more than this fraction
of the run is considered unbalanced."""

DISCLAIMER = (
    "Beating controls shows a performance difference under the tested conditions. "
    "It does not establish that the neurons learned."
)


@dataclass(frozen=True)
class Settings:
    real: str = "real"
    controls: tuple[str, ...] = ()
    alternative: stats.Alternative = "greater"
    alpha: float = 0.05
    looks: int = 1
    min_blocks: int = 3
    n_resamples: int = 10_000
    seed: int = 0


@dataclass
class CheckResult:
    id: str
    title: str
    status: str
    headline: str
    explanation: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Comparison:
    control: str
    n_real: int
    n_control: int
    mean_real: float
    mean_control: float
    difference: float
    interval: stats.Interval
    hedges_g: float
    p_value: float
    p_adjusted: float
    exact: bool
    min_p: float
    null: np.ndarray
    beaten: bool
    resolvable: bool


@dataclass
class Result:
    verdict: str
    verdict_text: str
    improvement: str
    checks: list[CheckResult]
    comparisons: list[Comparison]
    settings: Settings
    effective_alpha: float
    dataset: Dataset
    plan: PlanStatus | None

    def check(self, check_id: str) -> CheckResult:
        return next(c for c in self.checks if c.id == check_id)

    @property
    def exit_code(self) -> int:
        return {"beats-controls": 0, "inconclusive": 2}.get(self.verdict, 1)


def _fmt_p(p: float) -> str:
    if p < 1e-4:
        return "p < 0.0001"
    return f"p = {p:.4f}" if p < 0.01 else f"p = {p:.3f}"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def _improves(alternative: str) -> str:
    return {"greater": "increase", "less": "decrease"}.get(alternative, "change")


def _rngs(seed: int, n: int) -> list[np.random.Generator]:
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]


def resolve_controls(ds: Dataset, settings: Settings) -> tuple[str, ...]:
    conditions = ds.blocks.conditions
    if settings.real not in conditions:
        raise DataError(
            f"real condition {settings.real!r} not found; conditions are {conditions}"
        )
    controls = settings.controls or tuple(c for c in conditions if c != settings.real)
    missing = [c for c in controls if c not in conditions]
    if missing:
        raise DataError(f"control conditions {missing} not found; conditions are {conditions}")
    if not controls:
        raise DataError("no control condition: a result needs something to beat")
    return tuple(controls)


def _worst_case_values(ds: Dataset, mask: np.ndarray, bound: float) -> np.ndarray:
    """Block means if every missing measurement in these blocks had taken ``bound``."""
    b = ds.blocks
    measured, missing = b.n_measured[mask], b.n_missing[mask]
    total = np.nan_to_num(b.value[mask]) * measured + missing * bound
    return total / np.maximum(1, measured + missing)


def worst_case_comparisons(
    ds: Dataset, settings: Settings, comparisons: list[Comparison], alpha: float
) -> dict[str, bool]:
    """Would each beaten control still be beaten if every missing value went against
    the claim? Bounds are the smallest and largest measured values in the table."""
    b = ds.blocks
    measured = ds.trials.value if ds.trials is not None else b.value
    lo, hi = float(np.nanmin(measured)), float(np.nanmax(measured))
    strata = np.array(b.group) if b.group is not None else None
    rm = b.where(settings.real)
    raw = []
    rngs = _rngs(settings.seed + 3, len(comparisons))
    for comp, rng in zip(comparisons, rngs):
        cm = b.where(comp.control)
        up = settings.alternative == "greater" or (
            settings.alternative == "two-sided" and comp.difference >= 0)
        real_bound, ctrl_bound = (lo, hi) if up else (hi, lo)
        r = stats.mean_difference_test(
            _worst_case_values(ds, rm, real_bound), _worst_case_values(ds, cm, ctrl_bound),
            alternative=settings.alternative,
            strata_a=strata[rm] if strata is not None else None,
            strata_b=strata[cm] if strata is not None else None,
            n_resamples=settings.n_resamples, rng=rng,
        )
        raw.append(r.p_value)
    adjusted = stats.holm(raw)
    return {c.control: bool(c.beaten and p < alpha) for c, p in zip(comparisons, adjusted)}


def check_integrity(
    ds: Dataset,
    settings: Settings,
    controls: tuple[str, ...],
    comparisons: list[Comparison],
    alpha: float,
) -> CheckResult:
    b = ds.blocks
    per: dict[str, dict[str, int]] = {}
    for c in (settings.real, *controls):
        m = b.where(c)
        if ds.trial_level:
            missing, measured = int(b.n_missing[m].sum()), int(b.n_measured[m].sum())
        else:
            missing = int(np.isnan(b.value[m]).sum())
            measured = int(m.sum()) - missing
        per[c] = {
            "missing": missing,
            "measured": measured,
            "empty_blocks": int(np.isnan(b.value[m]).sum()),
        }
    unit = "trials" if ds.trial_level else "blocks"
    total_missing = sum(v["missing"] for v in per.values())
    total = sum(v["missing"] + v["measured"] for v in per.values())
    rate = {c: v["missing"] / max(1, v["missing"] + v["measured"]) for c, v in per.items()}

    raw = []
    for c in controls:
        r, k = per[settings.real], per[c]
        table = [[r["missing"], r["measured"]], [k["missing"], k["measured"]]]
        raw.append(1.0 if total_missing == 0 else float(sps.fisher_exact(table)[1]))
    adjusted = stats.holm(raw)
    differential = [(c, p) for c, p in zip(controls, adjusted) if p < settings.alpha]
    details: dict[str, Any] = {"unit": unit, "per_condition": per, "missing_rate": rate,
                               "fisher_p_adjusted": dict(zip(controls, adjusted))}
    if not total_missing:
        return CheckResult(
            "integrity", "Data integrity", PASS, f"No missing {unit}.",
            "Every block contributed a complete measurement.", details,
        )

    robust = worst_case_comparisons(ds, settings, comparisons, alpha)
    details["survives_worst_case"] = robust
    fragile = [c.control for c in comparisons if c.beaten and not robust[c.control]]
    explanation = (
        "Missing measurements are excluded from block means, never replaced with zero: "
        "a dropped frame is not a silent response. To see whether the conclusion depends "
        "on them, every missing value is set to the least favourable measured value (low "
        "for the real condition, high for controls) and the comparison is repeated."
    )
    summary = f"{total_missing} of {total} {unit} missing ({_pct(total_missing / total)})"
    if differential:
        c, p = min(differential, key=lambda x: x[1])
        gap = (f"{settings.real} lost {_pct(rate[settings.real])} vs {c} {_pct(rate[c])} "
               f"({_fmt_p(p)}, Holm)")
    if fragile and differential:
        return CheckResult(
            "integrity", "Data integrity", FAIL,
            f"Missing data differs between conditions ({gap}), and the advantage over "
            f"{', '.join(fragile)} does not survive worst-case values for it.",
            explanation + " When dropped frames correlate with outcome (for example, a "
            "stimulation artifact on difficult trials), survivorship alone can manufacture "
            "an advantage.", details,
        )
    if fragile:
        return CheckResult(
            "integrity", "Data integrity", WARN,
            f"{summary}. Rates are similar, but the advantage over {', '.join(fragile)} "
            "would not survive worst-case values for them.", explanation, details,
        )
    if differential:
        tail = ("the conclusion survives worst-case values for them."
                if any(c.beaten for c in comparisons) else "no control was beaten anyway.")
        return CheckResult(
            "integrity", "Data integrity", WARN, f"Missing data differs ({gap}), but {tail}",
            explanation, details,
        )
    tail = ("; the conclusion survives worst-case values for them."
            if any(c.beaten for c in comparisons) else "; rates do not differ detectably.")
    return CheckResult("integrity", "Data integrity", WARN, summary + tail, explanation,
                       details)


def compare_controls(
    ds: Dataset, settings: Settings, controls: tuple[str, ...], alpha: float
) -> tuple[CheckResult, list[Comparison]]:
    b = ds.blocks
    rngs = _rngs(settings.seed, 2 * len(controls))
    valid = ~np.isnan(b.value)
    real_mask = b.where(settings.real) & valid
    a = b.value[real_mask]
    strata = np.array(b.group) if b.group is not None else None

    raw: list[tuple] = []
    for i, c in enumerate(controls):
        ctrl_mask = b.where(c) & valid
        k = b.value[ctrl_mask]
        if a.size == 0 or k.size == 0:
            raise DataError(f"no usable blocks for {settings.real!r} or {c!r}")
        perm = stats.mean_difference_test(
            a, k, alternative=settings.alternative,
            strata_a=strata[real_mask] if strata is not None else None,
            strata_b=strata[ctrl_mask] if strata is not None else None,
            n_resamples=settings.n_resamples, rng=rngs[2 * i],
        )
        ci = stats.bootstrap_mean_difference(a, k, rng=rngs[2 * i + 1])
        raw.append((c, k, perm, ci))

    adjusted = stats.holm([r[2].p_value for r in raw])
    sign = {"greater": 1, "less": -1}.get(settings.alternative, 0)
    # Holm multiplies by up to m, so the design must resolve alpha / m.
    floor = alpha / len(controls)
    comparisons = []
    for (c, k, perm, ci), p_adj in zip(raw, adjusted):
        right_way = sign == 0 or np.sign(perm.observed) == sign
        comparisons.append(Comparison(
            control=c, n_real=a.size, n_control=k.size,
            mean_real=float(a.mean()), mean_control=float(k.mean()),
            difference=perm.observed, interval=ci, hedges_g=stats.hedges_g(a, k),
            p_value=perm.p_value, p_adjusted=p_adj, exact=perm.exact, min_p=perm.min_p,
            null=perm.null, beaten=bool(p_adj < alpha and right_way),
            resolvable=perm.min_p < floor and min(a.size, k.size) >= settings.min_blocks,
        ))

    beaten = [x for x in comparisons if x.beaten]
    unresolved = [x for x in comparisons if not x.beaten and not x.resolvable]
    n = len(comparisons)
    details = {"alpha": alpha, "n_controls": n}
    explanation = (
        "Each control is compared with the real condition using a permutation test on "
        "block means: condition labels are reshuffled between blocks"
        + (" within the same group" if strata is not None else "")
        + ", so the test assumes nothing about distributions. p-values are Holm-adjusted "
        "across controls, and every control must be beaten."
    )
    if len(beaten) == n:
        status, headline = PASS, f"{settings.real} beats all {n} control(s) at alpha = {alpha:g}."
    elif unresolved:
        worst = max(unresolved, key=lambda x: x.min_p)
        status = INCONCLUSIVE
        headline = (
            f"Too few blocks to decide: with {worst.n_real} vs {worst.n_control} blocks the "
            f"smallest achievable p is {worst.min_p:.3g}, above the {floor:.3g} needed."
        )
        explanation += (
            " A permutation test can only produce as many distinct p-values as there are "
            "relabellings, so small designs cannot reach significance however large the "
            "effect. Add blocks rather than trials."
        )
    else:
        losers = ", ".join(x.control for x in comparisons if not x.beaten)
        status = FAIL
        headline = f"{settings.real} does not beat: {losers} ({len(beaten)}/{n} beaten)."
    return CheckResult("controls", "Real vs controls", status, headline, explanation,
                       details), comparisons


def check_replication(
    ds: Dataset, settings: Settings, comparisons: list[Comparison], alpha: float
) -> CheckResult:
    if ds.trials is None:
        return CheckResult(
            "replication", "Unit of analysis", SKIPPED,
            "Block-level table: nothing to aggregate.",
            "Each row is already one block.",
        )
    t = ds.trials
    naive = {}
    for comp in comparisons:
        a = t.value[[c == settings.real for c in t.condition]]
        k = t.value[[c == comp.control for c in t.condition]]
        a, k = a[~np.isnan(a)], k[~np.isnan(k)]
        res = sps.ttest_ind(a, k, equal_var=False, alternative=settings.alternative)
        p = float(res.pvalue) if not math.isnan(res.pvalue) else 1.0
        naive[comp.control] = {"p_trial_level": p, "n_trials": int(a.size + k.size),
                               "p_block_level": comp.p_adjusted}
    misled = [c for c, v in naive.items() if v["p_trial_level"] < alpha
              and not next(x for x in comparisons if x.control == c).beaten]
    explanation = (
        "Trials inside one block share a culture state, a decoder state and a stretch of "
        "time, so they are not independent. Testing trials as if they were inflates the "
        "sample size and shrinks p-values (pseudoreplication). beatnull tests block means."
    )
    if misled:
        c = misled[0]
        v = naive[c]
        return CheckResult(
            "replication", "Unit of analysis", WARN,
            f"A trial-level t-test would claim {settings.real} beats {c} "
            f"({_fmt_p(v['p_trial_level'])}, n = {v['n_trials']} trials). "
            f"With blocks as the unit: {_fmt_p(v['p_block_level'])}.",
            explanation, {"naive": naive},
        )
    return CheckResult(
        "replication", "Unit of analysis", INFO,
        f"Trials averaged into {len(ds.blocks)} blocks; trial-level and block-level "
        "tests agree.", explanation, {"naive": naive},
    )


def check_order(ds: Dataset, settings: Settings, controls: tuple[str, ...]) -> CheckResult:
    b = ds.blocks
    if b.position is None:
        return CheckResult(
            "order", "Order effects", SKIPPED,
            "No position column: drift and schedule balance cannot be checked.",
            "Add the block's position in the run (0, 1, 2, ...) to check for drift.",
        )
    keep = ~np.isnan(b.value) & np.isin(np.array(b.condition), (settings.real, *controls))
    cond = np.array(b.condition)[keep]
    pos = b.position[keep]
    val = b.value[keep]
    resid = val.copy()
    for c in np.unique(cond):
        resid[cond == c] -= val[cond == c].mean()
    trend = stats.trend_test(pos, resid, alternative="two-sided",
                             n_resamples=settings.n_resamples, rng=_rngs(settings.seed + 1, 1)[0])
    span = float(pos.max() - pos.min()) or 1.0
    centre = float(pos.mean())
    mean_pos = {c: float(pos[cond == c].mean()) for c in np.unique(cond)}
    offsets = {c: (m - centre) / span for c, m in mean_pos.items()}
    worst = max(offsets, key=lambda c: abs(offsets[c]))
    imbalanced = abs(offsets[worst]) > IMBALANCE_LIMIT
    drifting = trend.p_value < settings.alpha
    details = {"rho": trend.observed, "p": trend.p_value, "mean_position": mean_pos,
               "offset": offsets}
    where = "later" if offsets[worst] > 0 else "earlier"
    if drifting and imbalanced:
        return CheckResult(
            "order", "Order effects", FAIL,
            f"Order confound: performance drifts across the run (rho = {trend.observed:+.2f}, "
            f"{_fmt_p(trend.p_value)}) and {worst} blocks sit "
            f"{_pct(abs(offsets[worst]))} {where} than average.",
            "When performance changes over the run (fatigue, culture drift, decoder "
            "adaptation) and conditions are not spread evenly across positions, the "
            "condition that ran later inherits the drift. Counterbalance the schedule, for "
            "example with a Latin square.",
            details,
        )
    if drifting:
        return CheckResult(
            "order", "Order effects", WARN,
            f"Performance drifts across the run (rho = {trend.observed:+.2f}, "
            f"{_fmt_p(trend.p_value)}), but the schedule is balanced.",
            "Drift affects all conditions; because positions are balanced it does not "
            "favour one condition, but it does add noise and may signal an unstable setup.",
            details,
        )
    if imbalanced:
        return CheckResult(
            "order", "Order effects", WARN,
            f"Unbalanced schedule: {worst} blocks sit {_pct(abs(offsets[worst]))} {where} "
            "than average. No drift detected.",
            "No drift was detected in this run, but an unbalanced schedule leaves the "
            "comparison exposed if drift is present and too small to detect.",
            details,
        )
    return CheckResult(
        "order", "Order effects", PASS,
        f"No detectable drift (rho = {trend.observed:+.2f}, {_fmt_p(trend.p_value)}); "
        "schedule balanced.",
        "Performance does not change detectably with position, and every condition is "
        "spread evenly across the run.",
        details,
    )


def check_improvement(ds: Dataset, settings: Settings, controls: tuple[str, ...]) -> CheckResult:
    b = ds.blocks
    if b.position is None:
        return CheckResult(
            "improvement", "Improvement over time", SKIPPED,
            "No position column.", "Learning-curve claims need block positions.",
        )
    valid = ~np.isnan(b.value)
    rm = b.where(settings.real) & valid
    cm = np.isin(np.array(b.condition), controls) & valid
    rng_a, rng_b, rng_c = _rngs(settings.seed + 2, 3)
    alt = settings.alternative
    real = stats.trend_test(b.position[rm], b.value[rm], alternative=alt,
                            n_resamples=settings.n_resamples, rng=rng_a)
    ctrl = stats.trend_test(b.position[cm], b.value[cm], alternative=alt,
                            n_resamples=settings.n_resamples, rng=rng_b)
    diff = stats.slope_difference_test(
        b.position[rm], b.value[rm], b.position[cm], b.value[cm],
        alternative=alt, n_resamples=settings.n_resamples, rng=rng_c,
    )
    details = {
        "real_rho": real.observed, "real_p": real.p_value,
        "control_rho": ctrl.observed, "control_p": ctrl.p_value,
        "slope_real": stats.slope(b.position[rm], b.value[rm]),
        "slope_controls": stats.slope(b.position[cm], b.value[cm]),
        "slope_difference_p": diff.p_value,
    }
    verb = _improves(alt)
    explanation = (
        "A learning curve in the real condition is only evidence of something specific to "
        "the real signal if the controls do not show the same curve. Decoders that adapt "
        "online, carried state and practice effects all improve controls too."
    )
    if real.p_value >= settings.alpha:
        return CheckResult(
            "improvement", "Improvement over time", INFO,
            f"No detectable {verb} across the run in {settings.real} "
            f"(rho = {real.observed:+.2f}, {_fmt_p(real.p_value)}).",
            explanation, details,
        )
    if diff.p_value < settings.alpha:
        return CheckResult(
            "improvement", "Improvement over time", PASS,
            f"{settings.real} improves across the run, faster than the controls "
            f"(slope difference {_fmt_p(diff.p_value)}).",
            explanation, details,
        )
    return CheckResult(
        "improvement", "Improvement over time", FAIL,
        f"{settings.real} improves across the run (rho = {real.observed:+.2f}, "
        f"{_fmt_p(real.p_value)}), but the controls improve just as much "
        f"(rho = {ctrl.observed:+.2f}). The improvement is not specific to the real "
        "condition.",
        explanation, details,
    )


def check_sequential(settings: Settings) -> tuple[CheckResult, float]:
    if settings.looks <= 1:
        return CheckResult(
            "sequential", "Interim looks", PASS,
            "Single planned analysis: nominal alpha applies.",
            "If you checked results before deciding to stop collecting, declare the number "
            "of looks (--looks): stopping when p dips below alpha inflates false positives.",
            {"looks": 1},
        ), settings.alpha
    design = stats.sequential_design(settings.looks, settings.alpha, settings.alternative)
    return CheckResult(
        "sequential", "Interim looks", INFO,
        f"{design.looks} looks: testing each at {settings.alpha:g} would give a "
        f"{_pct(design.naive_false_positive_rate)} false positive rate. Using a per-look "
        f"threshold of {design.per_look_alpha:.4f} instead.",
        "Each extra look is another chance for noise to cross the line. The per-look "
        "threshold is a constant (Pocock-type) boundary computed by simulating "
        "accumulating null data, which restores the overall false positive rate.",
        {"looks": design.looks, "naive_fpr": design.naive_false_positive_rate,
         "per_look_alpha": design.per_look_alpha},
    ), design.per_look_alpha


def check_plan(plan: PlanStatus | None) -> CheckResult:
    if plan is None:
        return CheckResult(
            "plan", "Pre-registration", WARN,
            "Exploratory analysis: no pre-registered plan.",
            "Without a plan frozen before the data existed, the metric, controls and "
            "direction may have been chosen after seeing results. Treat p-values as "
            "descriptive. Freeze one with `beatnull plan`.",
        )
    if not plan.intact:
        return CheckResult(
            "plan", "Pre-registration", FAIL,
            "Plan file was modified after it was frozen (hash mismatch).",
            f"Recorded {plan.recorded_sha256[:16]}..., computed "
            f"{plan.computed_sha256[:16]}.... Re-freeze it openly, and report the change.",
            {"path": plan.path},
        )
    return CheckResult(
        "plan", "Pre-registration", PASS,
        f"Analysed exactly as planned (plan {plan.computed_sha256[:12]}).",
        "The hash proves the plan has not changed since it was frozen; commit or "
        "timestamp the plan before collecting data to prove when that was.",
        {"path": plan.path, "sha256": plan.computed_sha256, "created": plan.plan.created},
    )


def run(ds: Dataset, settings: Settings | None = None,
        plan: PlanStatus | None = None) -> Result:
    """Run every check and combine them into a verdict."""
    settings = settings or Settings()
    controls = resolve_controls(ds, settings)
    sequential, alpha = check_sequential(settings)
    controls_check, comparisons = compare_controls(ds, settings, controls, alpha)
    integrity = check_integrity(ds, settings, controls, comparisons, alpha)
    checks = [
        check_plan(plan),
        integrity,
        controls_check,
        check_replication(ds, settings, comparisons, alpha),
        check_order(ds, settings, controls),
        check_improvement(ds, settings, controls),
        sequential,
    ]
    by_id = {c.id: c for c in checks}
    compromised = [c for c in ("plan", "integrity", "order") if by_id[c].status == FAIL]
    if compromised:
        verdict = "unreliable"
        text = "Unreliable: " + "; ".join(by_id[c].headline for c in compromised)
    elif controls_check.status == INCONCLUSIVE:
        verdict, text = "inconclusive", "Inconclusive. " + controls_check.headline
    elif controls_check.status == PASS:
        verdict, text = "beats-controls", controls_check.headline
    else:
        verdict, text = "does-not-beat-controls", controls_check.headline
    improvement = {PASS: "specific", FAIL: "not-specific", INFO: "none"}.get(
        by_id["improvement"].status, "unchecked")
    return Result(verdict, text, improvement, checks, comparisons, settings, alpha, ds, plan)
