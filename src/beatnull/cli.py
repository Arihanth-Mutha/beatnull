"""Command line: ``beatnull check | plan | demo | looks``.

Exit codes make it usable as a CI gate: 0 beats controls, 1 does not (or the
comparison is unreliable), 2 inconclusive, 3 bad input.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from . import __version__, check, demo
from .checks import FAIL, INCONCLUSIVE, PASS, WARN, Result
from .data import Columns, DataError, load
from .plan import Plan, PlanError, load_plan, save_plan
from .report import VERDICT_LABEL, write_html, write_json
from .stats import ALTERNATIVES, sequential_design

USE_COLOR = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
COLORS = {PASS: "32", FAIL: "31", WARN: "33", INCONCLUSIVE: "33"}


def _c(text: str, status: str) -> str:
    code = COLORS.get(status, "36")
    return f"\033[{code}m{text}\033[0m" if USE_COLOR else text


def _optional(value: str | None) -> str | None:
    return None if value in (None, "", "none", "-") else value


def _add_columns(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("columns")
    g.add_argument("--metric", default="value", help="metric column (default: value)")
    g.add_argument("--condition", default="condition", help="condition column")
    g.add_argument("--block", default="block", help="block id column; 'none' = one row per block")
    g.add_argument("--position", default="position", help="schedule position column")
    g.add_argument("--group", default="group",
                   help="stratum column (suite, culture); labels are only shuffled within it")


def _add_design(p: argparse.ArgumentParser) -> None:
    g = p.add_argument_group("design")
    g.add_argument("--real", default="real", help="name of the real condition")
    g.add_argument("--controls", default="", help="comma-separated controls (default: all others)")
    g.add_argument("--alternative", default="greater", choices=ALTERNATIVES,
                   help="direction of the claim: is real greater than controls?")
    g.add_argument("--alpha", type=float, default=0.05)
    g.add_argument("--looks", type=int, default=1,
                   help="how many times results were checked before stopping")
    g.add_argument("--min-blocks", type=int, default=3)


def _print(result: Result) -> None:
    label, _ = VERDICT_LABEL[result.verdict]
    status = PASS if result.verdict == "beats-controls" else (
        INCONCLUSIVE if result.verdict == "inconclusive" else FAIL)
    print(_c(f"■ {label.upper()}", status), "—", result.verdict_text)
    for ch in result.checks:
        print(f"  {_c(f'{ch.status:<12}', ch.status)} {ch.title:<22} {ch.headline}")
    for c in result.comparisons:
        print(f"    {result.settings.real} vs {c.control:<16} diff {c.difference:+.4f} "
              f"[{c.interval.low:+.4f}, {c.interval.high:+.4f}]  p_holm {c.p_adjusted:.4g}")


def cmd_check(args: argparse.Namespace) -> int:
    plan = load_plan(args.plan) if args.plan else None
    if plan is not None:
        p = plan.plan
        columns = Columns(p.metric, p.condition_column, p.block_column, p.position_column,
                          p.group_column)
    else:
        columns = Columns(args.metric, args.condition, _optional(args.block),
                          _optional(args.position), _optional(args.group))
    ds = load(args.data, columns)
    controls = tuple(c.strip() for c in args.controls.split(",") if c.strip())
    result = check(ds, real=args.real, controls=controls, alternative=args.alternative,
                   alpha=args.alpha, looks=args.looks, min_blocks=args.min_blocks,
                   n_resamples=args.resamples, seed=args.seed, plan=plan)
    _print(result)
    if args.report:
        print("report:", write_html(result, args.report))
    if args.json:
        print("json:  ", write_json(result, args.json))
    return result.exit_code


def cmd_plan(args: argparse.Namespace) -> int:
    plan = Plan(
        metric=args.metric, real=args.real,
        controls=tuple(c.strip() for c in args.controls.split(",") if c.strip()),
        alternative=args.alternative, alpha=args.alpha, looks=args.looks,
        min_blocks=args.min_blocks, condition_column=args.condition,
        block_column=_optional(args.block), position_column=_optional(args.position),
        group_column=_optional(args.group), note=args.note,
    )
    out = Path(args.output)
    if out.exists() and not args.force:
        print(f"{out} exists; a frozen plan is never overwritten silently (use --force)",
              file=sys.stderr)
        return 3
    digest = save_plan(plan, out)
    print(f"plan frozen: {out}\nsha256: {digest}")
    print("Commit or timestamp this file before collecting data.")
    return 0


def cmd_demo(args: argparse.Namespace) -> int:
    rows = demo.simulate(args.scenario, seed=args.seed)
    out = Path(args.output or f"{args.scenario}.csv")
    demo.write_csv(rows, out)
    print(f"wrote {len(rows)} trials to {out}")
    naive = demo.naive_learning_claim(rows)
    print(f"\nThe write-up version: real accuracy rose from {naive['early_accuracy']:.0%} "
          f"(first block) to {naive['late_accuracy']:.0%} (last block), "
          f"Welch t-test p = {naive['p_value']:.2g} on {naive['n_trials']} trials.\n")
    ds = load(out, Columns(metric="hit", group="suite"))
    result = check(ds, seed=args.seed)
    _print(result)
    if args.report:
        print("report:", write_html(result, args.report))
    return 0


def cmd_looks(args: argparse.Namespace) -> int:
    print(f"{'looks':>5}  {'false positive rate':>20}  {'per-look alpha':>14}")
    for k in range(1, args.max_looks + 1):
        d = sequential_design(k, args.alpha, args.alternative)
        print(f"{k:>5}  {d.naive_false_positive_rate:>20.1%}  {d.per_look_alpha:>14.4f}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="beatnull", description="Does your closed-loop result beat its controls?")
    parser.add_argument("--version", action="version", version=f"beatnull {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check", help="analyse a table of blocks or trials")
    p.add_argument("data", help="CSV, TSV, JSON or JSON Lines")
    p.add_argument("--plan", help="pre-registered plan (overrides design and column flags)")
    _add_columns(p)
    _add_design(p)
    p.add_argument("--resamples", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", help="write a standalone HTML report here")
    p.add_argument("--json", help="write machine-readable results here")
    p.set_defaults(func=cmd_check)

    p = sub.add_parser("plan", help="freeze an analysis plan before collecting data")
    _add_columns(p)
    _add_design(p)
    p.add_argument("--note", default="", help="free-text hypothesis or rationale")
    p.add_argument("-o", "--output", default="beatnull-plan.json")
    p.add_argument("--force", action="store_true")
    p.set_defaults(func=cmd_plan)

    p = sub.add_parser("demo", help="simulate a closed-loop experiment and check it")
    p.add_argument("scenario", choices=demo.SCENARIOS)
    p.add_argument("-o", "--output", help="CSV path (default: <scenario>.csv)")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--report", help="write a standalone HTML report here")
    p.set_defaults(func=cmd_demo)

    p = sub.add_parser("looks", help="false positive rate of peeking, by number of looks")
    p.add_argument("--max-looks", type=int, default=10)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--alternative", default="greater", choices=ALTERNATIVES)
    p.set_defaults(func=cmd_looks)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (DataError, PlanError, FileNotFoundError) as exc:
        print(f"beatnull: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
