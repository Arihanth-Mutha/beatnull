"""Pre-registered analysis plans.

Write the plan before collecting data: the primary metric, which condition is real,
which conditions are controls, the direction of the claim, alpha and how many times
you will look. ``beatnull check --plan`` then analyses the data exactly as planned
and reports any deviation.

The embedded SHA-256 detects a plan edited after freezing. It cannot prove *when* the
plan was written: commit it, or timestamp it externally, before the data exists.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from .stats import ALTERNATIVES

PLAN_FORMAT = 1


class PlanError(ValueError):
    """The plan file is malformed."""


@dataclass(frozen=True)
class Plan:
    metric: str
    real: str = "real"
    controls: tuple[str, ...] = ()
    alternative: str = "greater"
    alpha: float = 0.05
    looks: int = 1
    min_blocks: int = 3
    condition_column: str = "condition"
    block_column: str | None = "block"
    position_column: str | None = "position"
    group_column: str | None = "group"
    note: str = ""
    created: str = field(default_factory=lambda: datetime.now(UTC).isoformat(timespec="seconds"))

    def __post_init__(self) -> None:
        if self.alternative not in ALTERNATIVES:
            raise PlanError(f"alternative must be one of {ALTERNATIVES}")
        if not 0 < self.alpha < 1:
            raise PlanError("alpha must be between 0 and 1")
        if self.looks < 1:
            raise PlanError("looks must be at least 1")
        if self.real in self.controls:
            raise PlanError("the real condition cannot also be a control")


@dataclass(frozen=True)
class PlanStatus:
    plan: Plan
    path: str
    recorded_sha256: str
    computed_sha256: str

    @property
    def intact(self) -> bool:
        return self.recorded_sha256 == self.computed_sha256


def _canonical(body: dict) -> bytes:
    return json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _body(plan: Plan) -> dict:
    body = asdict(plan)
    body["controls"] = list(plan.controls)
    return {"beatnull_plan": PLAN_FORMAT, **body}


def plan_sha256(plan: Plan) -> str:
    return hashlib.sha256(_canonical(_body(plan))).hexdigest()


def save_plan(plan: Plan, path: str | Path) -> str:
    """Freeze a plan to disk and return its SHA-256."""
    digest = plan_sha256(plan)
    document = {**_body(plan), "plan_sha256": digest}
    Path(path).write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
    return digest


def load_plan(path: str | Path) -> PlanStatus:
    try:
        document = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise PlanError(f"{path} is not valid JSON: {exc}") from None
    if document.get("beatnull_plan") != PLAN_FORMAT:
        raise PlanError(f"{path} is not a beatnull plan (format {PLAN_FORMAT})")
    recorded = document.pop("plan_sha256", "")
    document.pop("beatnull_plan")
    known = set(Plan.__dataclass_fields__)
    unknown = set(document) - known
    if unknown:
        raise PlanError(f"{path} has unknown fields: {sorted(unknown)}")
    document["controls"] = tuple(document.get("controls", ()))
    plan = Plan(**document)
    return PlanStatus(plan, str(path), recorded, plan_sha256(plan))
