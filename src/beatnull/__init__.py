"""beatnull: does your closed-loop result beat its controls?

>>> import beatnull as bn
>>> ds = bn.load("runs.csv", metric="hit", group="suite")
>>> result = bn.check(ds, real="real", alternative="greater")
>>> result.verdict
'does-not-beat-controls'
>>> bn.write_html(result, "report.html")
"""

from __future__ import annotations

__version__ = "0.1.0"

from .checks import DISCLAIMER, Result, Settings  # noqa: E402
from .checks import run as _run  # noqa: E402
from .data import Columns, DataError, Dataset, from_rows, load  # noqa: E402
from .plan import Plan, PlanError, PlanStatus, load_plan, save_plan  # noqa: E402


def check(
    dataset: Dataset,
    *,
    real: str = "real",
    controls: tuple[str, ...] | list[str] = (),
    alternative: str = "greater",
    alpha: float = 0.05,
    looks: int = 1,
    min_blocks: int = 3,
    n_resamples: int = 10_000,
    seed: int = 0,
    plan: PlanStatus | None = None,
) -> Result:
    """Run every check on a loaded dataset. A plan, when given, overrides the settings."""
    if plan is not None:
        p = plan.plan
        real, controls, alternative = p.real, p.controls, p.alternative
        alpha, looks, min_blocks = p.alpha, p.looks, p.min_blocks
    settings = Settings(real, tuple(controls), alternative, alpha, looks, min_blocks,
                        n_resamples, seed)
    return _run(dataset, settings, plan)


from .report import render_html, to_dict, write_html, write_json  # noqa: E402

__all__ = [
    "DISCLAIMER", "Columns", "DataError", "Dataset", "Plan", "PlanError", "PlanStatus",
    "Result", "Settings", "__version__", "check", "from_rows", "load", "load_plan",
    "render_html", "save_plan", "to_dict", "write_html", "write_json",
]
