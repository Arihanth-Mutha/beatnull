import numpy as np
import pytest

import beatnull as bn
from beatnull import demo
from beatnull.checks import FAIL, INCONCLUSIVE, PASS, WARN
from beatnull.data import Columns, DataError, from_rows

DEMO_COLUMNS = Columns(metric="hit", group="suite")


@pytest.fixture(scope="module")
def scenarios():
    out = {}
    for name in demo.SCENARIOS:
        ds = from_rows(demo.simulate(name), DEMO_COLUMNS)
        out[name] = bn.check(ds, n_resamples=4000)
    return out


def test_noise_does_not_beat_controls(scenarios):
    r = scenarios["noise"]
    assert r.verdict == "does-not-beat-controls"
    assert r.exit_code == 1


def test_noise_learning_curve_is_not_specific(scenarios):
    r = scenarios["noise"]
    assert r.check("improvement").status == FAIL
    assert r.improvement == "not-specific"


def test_noise_naive_analysis_claims_learning():
    naive = demo.naive_learning_claim(demo.simulate("noise"))
    assert naive["late_accuracy"] > naive["early_accuracy"]
    assert naive["p_value"] < 0.05  # the trap the demo exists to show


def test_signal_beats_controls(scenarios):
    r = scenarios["signal"]
    assert r.verdict == "beats-controls"
    assert all(c.beaten for c in r.comparisons)
    assert r.exit_code == 0


def test_dropout_is_flagged_unreliable(scenarios):
    r = scenarios["dropout"]
    assert r.check("integrity").status == FAIL
    assert r.verdict == "unreliable"


@pytest.mark.slow
def test_demo_outcomes_are_not_cherry_picked():
    # The README's demo uses one seed; these rates hold across seeds.
    seeds = range(1, 11)
    verdicts = {name: [] for name in demo.SCENARIOS}
    for seed in seeds:
        for name in demo.SCENARIOS:
            ds = from_rows(demo.simulate(name, seed=seed), DEMO_COLUMNS)
            verdicts[name].append(bn.check(ds, n_resamples=1000, seed=seed).verdict)
    assert verdicts["noise"].count("does-not-beat-controls") >= 9
    assert verdicts["signal"].count("beats-controls") >= 9
    assert verdicts["dropout"].count("unreliable") >= 9


def test_trial_level_test_is_reported(scenarios):
    details = scenarios["noise"].check("replication").details["naive"]
    assert set(details) == {"zero", "random_matched"}


def _blocks(values, conditions, positions):
    return from_rows([
        {"block": f"b{i}", "condition": c, "value": v, "position": p}
        for i, (v, c, p) in enumerate(zip(values, conditions, positions))
    ])


def test_order_confound_is_unreliable():
    # Performance drifts upward, and every real block ran in the second half.
    n = 12
    positions = np.arange(2 * n)
    conditions = ["zero"] * n + ["real"] * n
    values = 0.02 * positions + np.random.default_rng(0).normal(0, 0.01, 2 * n)
    r = bn.check(_blocks(values, conditions, positions), n_resamples=2000)
    assert r.check("order").status == FAIL
    assert r.verdict == "unreliable"


def test_balanced_drift_is_only_a_warning():
    positions = np.arange(24)
    conditions = ["real", "zero"] * 12
    values = 0.02 * positions + np.random.default_rng(0).normal(0, 0.01, 24)
    r = bn.check(_blocks(values, conditions, positions), n_resamples=2000)
    assert r.check("order").status == WARN


def test_tiny_design_is_inconclusive_not_negative():
    r = bn.check(_blocks([0.9, 0.95, 0.1, 0.2], ["real", "real", "zero", "zero"], range(4)))
    assert r.verdict == "inconclusive"
    assert r.check("controls").status == INCONCLUSIVE
    assert r.exit_code == 2


def test_every_control_must_be_beaten():
    rng = np.random.default_rng(4)
    conditions = ["real", "zero", "shuffled"] * 10
    level = {"real": 1.0, "zero": 0.0, "shuffled": 1.0}
    values = [rng.normal(level[c], 0.1) for c in conditions]
    r = bn.check(_blocks(values, conditions, range(30)), n_resamples=2000)
    beaten = {c.control: c.beaten for c in r.comparisons}
    assert beaten == {"zero": True, "shuffled": False}
    assert r.verdict == "does-not-beat-controls"


def test_interim_looks_tighten_alpha():
    rng = np.random.default_rng(5)
    values = np.concatenate([rng.normal(0.3, 1, 12), rng.normal(0, 1, 12)])
    ds = _blocks(values, ["real"] * 12 + ["zero"] * 12, rng.permutation(24))
    r = bn.check(ds, looks=5, n_resamples=2000)
    assert r.effective_alpha < 0.05 / 2


def test_unknown_real_condition_is_an_input_error():
    with pytest.raises(DataError, match="not found"):
        bn.check(_blocks([1, 0], ["a", "b"], [0, 1]), real="real")


def test_passing_checks_say_so():
    rng = np.random.default_rng(6)
    conditions = ["real", "zero"] * 12
    values = [rng.normal(1 if c == "real" else 0, 0.2) for c in conditions]
    r = bn.check(_blocks(values, conditions, range(24)), n_resamples=2000)
    assert r.check("integrity").status == PASS
    assert r.check("order").status == PASS
    assert r.verdict == "beats-controls"
