"""Single-file HTML and JSON reports. No network, no external assets."""

from __future__ import annotations

import json
import math
import platform
from datetime import UTC, datetime
from html import escape
from pathlib import Path
from typing import Any

import numpy as np

from . import __version__, svg
from .checks import DISCLAIMER, Result

VERDICT_LABEL = {
    "beats-controls": ("Beats controls", "pass"),
    "does-not-beat-controls": ("Does not beat controls", "fail"),
    "inconclusive": ("Inconclusive", "inconclusive"),
    "unreliable": ("Unreliable comparison", "fail"),
}
IMPROVEMENT_LABEL = {
    "specific": "Improvement over time is specific to the real condition.",
    "not-specific": "Improvement over time is not specific: the controls improve too.",
    "none": "No detectable improvement over time.",
    "unchecked": "Improvement over time not checked (no position column).",
}

CSS = """
:root{--bg:#fbfaf7;--panel:#fff;--ink:#1d1f23;--muted:#62666d;--line:#e4e1da;
--real:#2f6fdb;--c1:#d9822b;--c2:#8a63d2;--c3:#5b8c5a;--pass:#1f7a4d;--pass-bg:#e5f4ec;
--fail:#b4232c;--fail-bg:#fbe9ea;--warn:#93620b;--warn-bg:#fbf1dc;--info:#3c5a8a;
--info-bg:#e8eef8;--tail:#b4232c}
@media (prefers-color-scheme:dark){:root:not([data-theme=light]){--bg:#15171a;--panel:#1d2024;
--ink:#e8e6e1;--muted:#9da1a8;--line:#30343a;--real:#6ea1ff;--c1:#f0a35c;--c2:#b495f0;
--c3:#8cc08a;--pass:#6fd09c;--pass-bg:#173326;--fail:#ff8a91;--fail-bg:#3a1d20;--warn:#f0c26a;
--warn-bg:#352a14;--info:#9cb8ea;--info-bg:#1d2738;--tail:#ff8a91}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font:15px/1.55 ui-sans-serif,-apple-system,"Segoe UI",Roboto,sans-serif}
main{max-width:920px;margin:0 auto;padding:32px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:36px 0 12px}
.meta{color:var(--muted);font-size:13px;word-break:break-all}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.5px}
.verdict{margin:24px 0;padding:20px 22px;border-radius:12px;border:1px solid var(--line);
background:var(--panel)}
.verdict .label{font-size:26px;font-weight:700;letter-spacing:-.01em}
.verdict.pass .label{color:var(--pass)}.verdict.fail .label{color:var(--fail)}
.verdict.inconclusive .label{color:var(--warn)}
.verdict p{margin:6px 0 0}.disclaimer{color:var(--muted);font-size:13px;margin-top:10px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:14px 16px;
margin:10px 0}.card h3{font-size:15px;margin:0 0 4px;display:flex;gap:10px;align-items:center}
.card p{margin:4px 0 0}.card .why{color:var(--muted);font-size:13.5px}
.pill{font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;
padding:2px 8px;border-radius:99px}
.pill.pass{color:var(--pass);background:var(--pass-bg)}.pill.fail{color:var(--fail);
background:var(--fail-bg)}.pill.warn,.pill.inconclusive{color:var(--warn);background:var(--warn-bg)}
.pill.info,.pill.skipped{color:var(--info);background:var(--info-bg)}
.table-wrap{overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13.5px;
background:var(--panel);border:1px solid var(--line);border-radius:10px}
th,td{text-align:right;padding:8px 10px;border-bottom:1px solid var(--line);white-space:nowrap}
th:first-child,td:first-child{text-align:left}th{color:var(--muted);font-weight:600}
tr:last-child td{border-bottom:0}
figure{margin:12px 0;background:var(--panel);border:1px solid var(--line);border-radius:10px;
padding:10px}figcaption{color:var(--muted);font-size:13px;padding:4px 6px 0}
svg{width:100%;height:auto;display:block}
.grid{stroke:var(--line)}.tick,.cat,.legend{fill:var(--muted);font-size:11px}
.axis-label{fill:var(--muted);font-size:12px}
.dot{fill-opacity:.75}.dot.faint{fill-opacity:.28}
.real{fill:var(--real);stroke:var(--real)}.c1{fill:var(--c1);stroke:var(--c1)}
.c2{fill:var(--c2);stroke:var(--c2)}.c3{fill:var(--c3);stroke:var(--c3)}
.dot{stroke:none}.mean{stroke:var(--ink);stroke-width:2.5}.ci{stroke:var(--ink);stroke-width:1.5}
.trend{fill:none;stroke-width:2.5}.swatch{stroke:none}
.bar{fill:var(--muted);fill-opacity:.45}.bar.tail{fill:var(--tail);fill-opacity:.7}
.observed{stroke:var(--ink);stroke-width:2;stroke-dasharray:4 3}
.yes{color:var(--pass);font-weight:600}.no{color:var(--fail);font-weight:600}
"""


def _p(p: float) -> str:
    return "&lt;0.0001" if p < 1e-4 else f"{p:.4f}"


def _num(x: float, digits: int = 3) -> str:
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def _mean_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    if values.size < 2:
        v = float(values.mean()) if values.size else 0.0
        return v, v
    rng = np.random.default_rng(seed)
    means = values[rng.integers(0, values.size, (4000, values.size))].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975])
    return float(lo), float(hi)


def render_html(result: Result) -> str:
    ds, s, b = result.dataset, result.settings, result.dataset.blocks
    label, tone = VERDICT_LABEL[result.verdict]
    metric = ds.columns.metric
    order = (s.real, *[c.control for c in result.comparisons])
    valid = ~np.isnan(b.value)

    groups = []
    for i, c in enumerate(order):
        vals = b.value[b.where(c) & valid]
        groups.append((c, vals, _mean_interval(vals, s.seed + i)))

    rows = []
    for c in result.comparisons:
        rows.append(
            f"<tr><td>{escape(c.control)}</td><td>{c.n_real} / {c.n_control}</td>"
            f"<td>{_num(c.mean_real)}</td><td>{_num(c.mean_control)}</td>"
            f"<td>{c.difference:+.3f} [{c.interval.low:+.3f}, {c.interval.high:+.3f}]</td>"
            f"<td>{_num(c.hedges_g, 2)}</td><td>{_p(c.p_value)}</td><td>{_p(c.p_adjusted)}</td>"
            f"<td class=\"{'yes' if c.beaten else 'no'}\">{'beaten' if c.beaten else 'not beaten'}"
            "</td></tr>"
        )

    figures = [
        f"<figure>{svg.strip_plot(groups, metric)}<figcaption>Each dot is one block's mean "
        f"{escape(metric)}. Bars show the condition mean and its 95% bootstrap interval."
        "</figcaption></figure>"
    ]
    if b.position is not None:
        series = []
        for c in order:
            m = b.where(c) & valid
            series.append((c, b.position[m], b.value[m]))
        figures.append(
            f"<figure>{svg.line_plot(series, 'position in run', metric)}<figcaption>Mean "
            f"{escape(metric)} at each schedule position. If the controls rise in parallel "
            "with the real condition, the rise comes from drift or decoder adaptation, not "
            "from the real signal.</figcaption></figure>"
        )
    for c in result.comparisons:
        hist = svg.null_histogram(c.null, c.difference, s.alternative,
                                  f"mean difference vs {c.control}")
        figures.append(
            f"<figure>{hist}<figcaption>Differences produced by reshuffling condition labels "
            f"({c.null.size:,} {'enumerated' if c.exact else 'sampled'} relabellings). "
            f"Red bars are at least as extreme as the observed difference; their share is the "
            f"unadjusted p-value ({_p(c.p_value)}).</figcaption></figure>"
        )

    cards = []
    for ch in result.checks:
        cards.append(
            f'<div class="card"><h3><span class="pill {ch.status}">{ch.status}</span>'
            f"{escape(ch.title)}</h3><p>{escape(ch.headline)}</p>"
            f'<p class="why">{escape(ch.explanation)}</p></div>'
        )

    plan_line = (
        f"plan <span class=mono>{escape(result.plan.path)}</span> "
        f"(<span class=mono>{result.plan.computed_sha256[:16]}</span>)"
        if result.plan else "no pre-registered plan (exploratory)"
    )
    generated = datetime.now(UTC).isoformat(timespec="seconds")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>beatnull report</title><style>{CSS}</style></head>
<body><main>
<h1>beatnull report</h1>
<div class="meta">{escape(ds.source)} · sha256 <span class=mono>{ds.sha256[:16]}</span> ·
{len(b)} blocks{' from trial-level rows' if ds.trial_level else ''} · {plan_line}</div>

<section class="verdict {tone}">
<div class="label">{label}</div>
<p>{escape(result.verdict_text)}</p>
<p>{IMPROVEMENT_LABEL[result.improvement]}</p>
<p class="disclaimer">{escape(DISCLAIMER)}</p>
</section>

<h2>Real vs each control</h2>
<div class="table-wrap"><table>
<tr><th>control</th><th>blocks</th><th>real</th><th>control</th>
<th>difference [95% CI]</th><th>Hedges g</th><th>p</th><th>p (Holm)</th><th>result</th></tr>
{''.join(rows)}
</table></div>
<p class="meta">Metric <code>{escape(metric)}</code>, alternative <code>{s.alternative}</code>,
alpha {result.effective_alpha:.4g}{' (adjusted for interim looks)' if s.looks > 1 else ''}.
Blocks are the unit of analysis; p-values come from block-label permutations.</p>

<h2>The evidence</h2>
{''.join(figures)}

<h2>Checks</h2>
{''.join(cards)}

<p class="meta">beatnull {__version__} · python {platform.python_version()} · seed {s.seed} ·
{s.n_resamples:,} resamples · generated {generated}</p>
</main></body></html>
"""


def to_dict(result: Result) -> dict[str, Any]:
    def clean(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(k): clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, (np.floating, float)):
            return None if math.isnan(value) else float(value)
        if isinstance(value, np.integer):
            return int(value)
        return value

    return clean({
        "beatnull": __version__,
        "verdict": result.verdict,
        "verdict_text": result.verdict_text,
        "improvement": result.improvement,
        "disclaimer": DISCLAIMER,
        "source": result.dataset.source,
        "source_sha256": result.dataset.sha256,
        "plan_sha256": result.plan.computed_sha256 if result.plan else None,
        "settings": {**result.settings.__dict__, "controls": list(result.settings.controls)},
        "effective_alpha": result.effective_alpha,
        "comparisons": [
            {"control": c.control, "n_real": c.n_real, "n_control": c.n_control,
             "mean_real": c.mean_real, "mean_control": c.mean_control,
             "difference": c.difference, "ci95": [c.interval.low, c.interval.high],
             "hedges_g": c.hedges_g, "p": c.p_value, "p_holm": c.p_adjusted,
             "exact": c.exact, "min_p": c.min_p, "beaten": c.beaten}
            for c in result.comparisons
        ],
        "checks": [
            {"id": c.id, "status": c.status, "headline": c.headline, "details": c.details}
            for c in result.checks
        ],
    })


def write_html(result: Result, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(render_html(result), encoding="utf-8")
    return path


def write_json(result: Result, path: str | Path) -> Path:
    path = Path(path)
    path.write_text(json.dumps(to_dict(result), indent=2) + "\n", encoding="utf-8")
    return path
