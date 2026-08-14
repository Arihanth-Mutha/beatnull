"""Load experiment tables and reduce them to one row per block.

Rows may be blocks (one value each) or trials (many rows share a block id). Trials
are averaged within their block, because trials inside one block share a culture
state, a decoder state and a stretch of time: they are not independent samples.

Missing values stay missing. An empty cell, ``nan``, ``NA`` or ``null`` is counted
as a dropped measurement and never replaced with zero.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MISSING_TOKENS = frozenset({"", "nan", "na", "n/a", "null", "none", "missing"})


class DataError(ValueError):
    """The table cannot be analysed as given."""


@dataclass(frozen=True)
class Columns:
    metric: str = "value"
    condition: str = "condition"
    block: str | None = "block"
    position: str | None = "position"
    group: str | None = "group"


@dataclass(frozen=True)
class Blocks:
    """One entry per experimental block, in input order."""

    key: tuple[str, ...]
    condition: tuple[str, ...]
    value: np.ndarray
    """Block mean of the metric; NaN when the block has no usable measurement."""
    n_measured: np.ndarray
    n_missing: np.ndarray
    position: np.ndarray | None
    group: tuple[str, ...] | None

    def __len__(self) -> int:
        return len(self.key)

    @property
    def conditions(self) -> list[str]:
        return list(dict.fromkeys(self.condition))

    def where(self, condition: str) -> np.ndarray:
        return np.array([c == condition for c in self.condition], dtype=bool)


@dataclass(frozen=True)
class Trials:
    """Trial-level values, kept only to show what a naive trial-level test would say."""

    condition: tuple[str, ...]
    value: np.ndarray


@dataclass(frozen=True)
class Dataset:
    blocks: Blocks
    trials: Trials | None
    columns: Columns
    source: str
    sha256: str

    @property
    def trial_level(self) -> bool:
        return self.trials is not None


def _parse_number(raw: object, *, column: str, row: int) -> float:
    if raw is None:
        return math.nan
    if isinstance(raw, bool):
        return float(raw)
    if isinstance(raw, (int, float)):
        return float(raw)
    text = str(raw).strip()
    if text.lower() in MISSING_TOKENS:
        return math.nan
    if text.lower() in {"true", "false"}:
        return 1.0 if text.lower() == "true" else 0.0
    try:
        return float(text)
    except ValueError:
        raise DataError(f"row {row}: column {column!r} has non-numeric value {text!r}") from None


def _text(raw: object) -> str:
    return "" if raw is None else str(raw).strip()


def read_rows(path: str | Path) -> tuple[list[dict[str, object]], str]:
    """Read CSV, TSV, JSON (a list of objects) or JSON Lines; return rows and a SHA-256."""
    path = Path(path)
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    text = data.decode("utf-8-sig")
    suffix = path.suffix.lower()
    if suffix == ".json":
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise DataError("a .json file must contain a list of row objects")
        return parsed, digest
    if suffix in {".jsonl", ".ndjson"}:
        return [json.loads(line) for line in text.splitlines() if line.strip()], digest
    dialect = "excel-tab" if suffix == ".tsv" else "excel"
    return list(csv.DictReader(io.StringIO(text), dialect=dialect)), digest


def load(path: str | Path, columns: Columns | None = None, **overrides: str | None) -> Dataset:
    """Load a table from disk. Keyword overrides replace individual column names."""
    columns = columns or Columns()
    if overrides:
        columns = Columns(**{**columns.__dict__, **overrides})
    rows, digest = read_rows(path)
    return from_rows(rows, columns, source=str(path), sha256=digest)


def from_rows(
    rows: Iterable[Mapping[str, object]],
    columns: Columns | None = None,
    *,
    source: str = "<memory>",
    sha256: str | None = None,
) -> Dataset:
    columns = columns or Columns()
    rows = list(rows)
    if not rows:
        raise DataError("the table has no rows")
    header = set(rows[0].keys())
    for required in (columns.metric, columns.condition):
        if required not in header:
            raise DataError(
                f"column {required!r} not found; available columns: {sorted(header)}"
            )

    def present(name: str | None) -> str | None:
        return name if name is not None and name in header else None

    block_col = present(columns.block)
    position_col = present(columns.position)
    group_col = present(columns.group)
    resolved = Columns(columns.metric, columns.condition, block_col, position_col, group_col)

    order: list[str] = []
    cond: dict[str, str] = {}
    pos: dict[str, float] = {}
    grp: dict[str, str] = {}
    values: dict[str, list[float]] = {}
    trial_cond: list[str] = []
    trial_val: list[float] = []

    for i, row in enumerate(rows, start=1):  # 1-based data row, header excluded
        condition = _text(row.get(columns.condition))
        if not condition:
            raise DataError(f"row {i}: empty {columns.condition!r}")
        group = _text(row.get(group_col)) if group_col else ""
        block = _text(row.get(block_col)) if block_col else str(i)
        if block_col and not block:
            raise DataError(f"row {i}: empty {block_col!r}")
        key = f"{group}/{block}" if group_col else block
        value = _parse_number(row.get(columns.metric), column=columns.metric, row=i)
        position = (
            _parse_number(row.get(position_col), column=position_col, row=i)
            if position_col else math.nan
        )
        if key not in values:
            order.append(key)
            values[key] = []
            cond[key] = condition
            pos[key] = position
            grp[key] = group
        else:
            if cond[key] != condition:
                raise DataError(
                    f"row {i}: block {key!r} appears under conditions "
                    f"{cond[key]!r} and {condition!r}"
                )
            if position_col and not (
                pos[key] == position or (math.isnan(pos[key]) and math.isnan(position))
            ):
                raise DataError(f"row {i}: block {key!r} has more than one {position_col!r}")
        values[key].append(value)
        trial_cond.append(condition)
        trial_val.append(value)

    n_measured = np.array([sum(not math.isnan(v) for v in values[k]) for k in order])
    n_missing = np.array([sum(math.isnan(v) for v in values[k]) for k in order])
    with np.errstate(invalid="ignore"):
        means = np.array(
            [np.nanmean(values[k]) if n else math.nan for k, n in zip(order, n_measured)],
            dtype=float,
        )
    positions = np.array([pos[k] for k in order], dtype=float) if position_col else None
    if positions is not None and np.isnan(positions).any():
        raise DataError(f"every block needs a {position_col!r} value")

    blocks = Blocks(
        key=tuple(order),
        condition=tuple(cond[k] for k in order),
        value=means,
        n_measured=n_measured,
        n_missing=n_missing,
        position=positions,
        group=tuple(grp[k] for k in order) if group_col else None,
    )
    trial_level = block_col is not None and len(trial_val) > len(order)
    trials = Trials(tuple(trial_cond), np.array(trial_val, dtype=float)) if trial_level else None
    if sha256 is None:
        canonical = json.dumps([dict(r) for r in rows], sort_keys=True, default=str)
        sha256 = hashlib.sha256(canonical.encode()).hexdigest()
    return Dataset(blocks, trials, resolved, source, sha256)
