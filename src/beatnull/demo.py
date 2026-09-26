"""Synthetic closed-loop experiments that look convincing and are not.

A two-target task: each trial a target appears on the left or right (right 70% of
the time). The encoder stimulates the electrode group for the target, an
8-channel "culture" responds with Poisson spike counts, and an online logistic
decoder carried across blocks turns counts into an action. Three conditions run
in a Latin-square schedule inside each suite:

- ``real``            stimulation encodes the target
- ``zero``            no stimulation
- ``random_matched``  stimulation with the same statistics, unrelated to the target

Scenarios:

- ``noise``    the culture ignores stimulation. Accuracy still climbs from 50% to ~70%
               in every condition, because the decoder learns the 70/30 target bias.
               A first-block vs last-block test on the real condition says "learning".
- ``signal``   stimulated channels fire more. Real genuinely beats both controls.
- ``dropout``  the culture ignores stimulation, but in the real condition a
               stimulation artifact drops frames, more often on trials the decoder
               would have missed. Real appears to win; only the missing data shows why.

Nothing here models biology. It models the analysis traps.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

SCENARIOS = ("noise", "signal", "dropout")
CONDITIONS = ("real", "zero", "random_matched")
LATIN_ROWS = ((0, 1, 2), (1, 2, 0), (2, 0, 1))

N_CHANNELS = 8
BASE_RATE = 2.0
TARGET_BIAS = 0.7


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def simulate(
    scenario: str = "noise",
    *,
    suites: int = 4,
    cycles: int = 3,
    trials: int = 60,
    gain: float = 0.8,
    learning_rate: float = 0.01,
    bias_rate: float = 0.006,
    decay: float = 0.4,
    drop_rate: float = 0.01,
    seed: int = 1,
) -> list[dict[str, object]]:
    """Return trial-level rows: suite, block, position, condition, trial, hit.

    The decoder starts from noisy channel weights (as if fitted on a short, noisy
    calibration), shrinks them with L2 decay, and slowly learns the target bias. So
    accuracy drifts from 50% toward 70% in every condition with no neural signal at all.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f"scenario must be one of {SCENARIOS}")
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for suite in range(1, suites + 1):
        # Decoder state is carried across the suite's blocks (a "carry" policy).
        weights = rng.normal(0.0, 0.5, N_CHANNELS)
        bias = 0.0
        schedule: list[str] = []
        for _ in range(cycles):
            row = LATIN_ROWS[rng.integers(len(LATIN_ROWS))]
            schedule += [CONDITIONS[i] for i in row]
        for position, condition in enumerate(schedule):
            block = f"s{suite}-b{position + 1:02d}"
            for trial in range(trials):
                target = int(rng.random() < TARGET_BIAS)
                if condition == "real":
                    stim: int | None = target
                elif condition == "random_matched":
                    stim = int(rng.random() < TARGET_BIAS)
                else:
                    stim = None
                rates = np.full(N_CHANNELS, BASE_RATE)
                if scenario == "signal" and stim is not None:
                    half = N_CHANNELS // 2
                    rates[stim * half:(stim + 1) * half] *= 1 + gain
                counts = rng.poisson(rates)
                x = (counts - BASE_RATE) / math.sqrt(BASE_RATE)
                score = float(weights @ x) + bias
                p = _sigmoid(score)
                action = int(score > 0) if score != 0 else int(rng.random() < 0.5)
                hit = int(action == target)
                # Online update after every trial, in every condition.
                weights += learning_rate * ((target - p) * x - decay * weights)
                bias += bias_rate * (target - p)

                dropped = rng.random() < drop_rate
                if scenario == "dropout" and condition == "real":
                    dropped = dropped or (not hit and rng.random() < 0.5)
                rows.append({
                    "suite": f"s{suite}", "block": block, "position": position,
                    "condition": condition, "trial": trial,
                    "hit": "" if dropped else hit,
                })
    return rows


def write_csv(rows: list[dict[str, object]], path: str | Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def naive_learning_claim(rows: list[dict[str, object]], *, condition: str = "real") -> dict:
    """The analysis that gets written up: each suite's first vs last real block, with
    every trial treated as an independent sample and no look at the controls."""
    from scipy import stats as sps

    blocks: dict[str, list[tuple[int, str]]] = {}
    for r in rows:
        if r["condition"] == condition:
            entry = (int(r["position"]), str(r["block"]))
            suite_blocks = blocks.setdefault(str(r["suite"]), [])
            if entry not in suite_blocks:
                suite_blocks.append(entry)
    first = {min(v)[1] for v in blocks.values()}
    last = {max(v)[1] for v in blocks.values()}

    def hits(chosen: set[str]) -> np.ndarray:
        return np.array([float(r["hit"]) for r in rows
                         if r["block"] in chosen and r["hit"] != ""])

    early, late = hits(first), hits(last)
    t = sps.ttest_ind(late, early, equal_var=False, alternative="greater")
    return {"early_accuracy": float(early.mean()), "late_accuracy": float(late.mean()),
            "n_trials": int(early.size + late.size), "p_value": float(t.pvalue)}
