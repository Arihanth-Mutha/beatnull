"""Dependency-free inline SVG charts, styled by the report's CSS variables."""

from __future__ import annotations

import math
from collections.abc import Sequence
from html import escape

import numpy as np

W, H = 640, 300
PAD_L, PAD_R, PAD_T, PAD_B = 56, 16, 16, 44


def nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not math.isfinite(lo) or not math.isfinite(hi):
        return [0.0, 1.0]
    if hi <= lo:
        hi = lo + (abs(lo) or 1.0)
    raw = (hi - lo) / n
    mag = 10 ** math.floor(math.log10(raw))
    step = next(s * mag for s in (1, 2, 2.5, 5, 10) if s * mag >= raw)
    start = math.floor(lo / step) * step
    ticks = []
    t = start
    while t <= hi + step * 0.5:
        ticks.append(round(t, 12))
        t += step
    return ticks


def _fmt(v: float) -> str:
    if v == 0:
        return "0"
    if abs(v) >= 100 or float(v).is_integer():
        return f"{v:.0f}"
    return f"{v:.3g}"


class Frame:
    def __init__(self, x_lo: float, x_hi: float, y_lo: float, y_hi: float):
        self.yt = nice_ticks(y_lo, y_hi)
        self.y_lo, self.y_hi = self.yt[0], self.yt[-1]
        self.x_lo, self.x_hi = x_lo, x_hi if x_hi > x_lo else x_lo + 1

    def x(self, v: float) -> float:
        return PAD_L + (v - self.x_lo) / (self.x_hi - self.x_lo) * (W - PAD_L - PAD_R)

    def y(self, v: float) -> float:
        span = (self.y_hi - self.y_lo) or 1.0
        return H - PAD_B - (v - self.y_lo) / span * (H - PAD_T - PAD_B)

    def grid(self, y_label: str) -> str:
        parts = []
        for t in self.yt:
            yy = self.y(t)
            parts.append(f'<line class="grid" x1="{PAD_L}" x2="{W - PAD_R}" y1="{yy:.1f}" '
                         f'y2="{yy:.1f}"/><text class="tick" x="{PAD_L - 8}" y="{yy + 4:.1f}" '
                         f'text-anchor="end">{_fmt(t)}</text>')
        parts.append(f'<text class="axis-label" transform="translate(14 {(H - PAD_B + PAD_T) / 2})'
                     f' rotate(-90)" text-anchor="middle">{escape(y_label)}</text>')
        return "".join(parts)


def _svg(body: str, label: str) -> str:
    return (f'<svg viewBox="0 0 {W} {H}" role="img" aria-label="{escape(label)}" '
            f'xmlns="http://www.w3.org/2000/svg">{body}</svg>')


def strip_plot(
    groups: Sequence[tuple[str, np.ndarray, tuple[float, float]]], y_label: str
) -> str:
    """One column per condition: every block as a dot, the mean with its 95% interval."""
    values = np.concatenate([g[1] for g in groups]) if groups else np.array([0.0])
    lows = [g[2][0] for g in groups] + [float(values.min())]
    highs = [g[2][1] for g in groups] + [float(values.max())]
    f = Frame(-0.5, len(groups) - 0.5, min(lows), max(highs))
    parts = [f.grid(y_label)]
    for i, (name, vals, (lo, hi)) in enumerate(groups):
        cx = f.x(i)
        series = "real" if i == 0 else f"c{min(i, 3)}"
        for j, v in enumerate(vals):
            jitter = ((j * 0.618034) % 1 - 0.5) * 70
            parts.append(f'<circle class="dot {series}" cx="{cx + jitter:.1f}" '
                         f'cy="{f.y(v):.1f}" r="4"/>')
        mean = float(vals.mean())
        parts.append(f'<line class="ci" x1="{cx:.1f}" x2="{cx:.1f}" y1="{f.y(lo):.1f}" '
                     f'y2="{f.y(hi):.1f}"/>')
        parts.append(f'<line class="mean" x1="{cx - 48:.1f}" x2="{cx + 48:.1f}" '
                     f'y1="{f.y(mean):.1f}" y2="{f.y(mean):.1f}"/>')
        parts.append(f'<text class="cat" x="{cx:.1f}" y="{H - PAD_B + 20}" '
                     f'text-anchor="middle">{escape(name)} (n={len(vals)})</text>')
    return _svg("".join(parts), f"{y_label} per block by condition")


def line_plot(
    series: Sequence[tuple[str, np.ndarray, np.ndarray]], x_label: str, y_label: str
) -> str:
    """Mean value at each position, one line per condition, with blocks as faint dots."""
    xs = np.concatenate([s[1] for s in series])
    ys = np.concatenate([s[2] for s in series])
    f = Frame(float(xs.min()), float(xs.max()), float(ys.min()), float(ys.max()))
    parts = [f.grid(y_label)]
    for t in nice_ticks(float(xs.min()), float(xs.max()), 8):
        if f.x_lo <= t <= f.x_hi:
            parts.append(f'<text class="tick" x="{f.x(t):.1f}" y="{H - PAD_B + 18}" '
                         f'text-anchor="middle">{_fmt(t)}</text>')
    parts.append(f'<text class="axis-label" x="{(W + PAD_L) / 2}" y="{H - 6}" '
                 f'text-anchor="middle">{escape(x_label)}</text>')
    legend = []
    for i, (name, x, y) in enumerate(series):
        cls = "real" if i == 0 else f"c{min(i, 3)}"
        for xi, yi in zip(x, y):
            parts.append(f'<circle class="dot faint {cls}" cx="{f.x(xi):.1f}" '
                         f'cy="{f.y(yi):.1f}" r="2.5"/>')
        ux = np.unique(x)
        means = [float(y[x == u].mean()) for u in ux]
        pts = " ".join(f"{f.x(u):.1f},{f.y(m):.1f}" for u, m in zip(ux, means))
        parts.append(f'<polyline class="trend {cls}" points="{pts}"/>')
        lx = PAD_L + 12 + i * 150
        legend.append(f'<rect class="swatch {cls}" x="{lx}" y="{PAD_T}" width="12" height="12" '
                      f'rx="2"/><text class="legend" x="{lx + 18}" y="{PAD_T + 10}">'
                      f'{escape(name)}</text>')
    return _svg("".join(parts + legend), f"{y_label} across {x_label}")


def null_histogram(null: np.ndarray, observed: float, alternative: str, label: str) -> str:
    """Permutation null distribution with the observed statistic marked."""
    lo = float(min(null.min(), observed))
    hi = float(max(null.max(), observed))
    pad = (hi - lo) * 0.05 or 0.01
    edges = np.linspace(lo - pad, hi + pad, 41)
    counts, _ = np.histogram(null, bins=edges)
    f = Frame(edges[0], edges[-1], 0, float(counts.max()) * 1.08)
    parts = [f.grid("relabellings")]
    for c, a, b in zip(counts, edges[:-1], edges[1:]):
        mid = (a + b) / 2
        tail = (
            (alternative == "greater" and mid >= observed)
            or (alternative == "less" and mid <= observed)
            or (alternative == "two-sided" and abs(mid) >= abs(observed))
        )
        x0, x1 = f.x(a), f.x(b)
        parts.append(f'<rect class="bar{" tail" if tail else ""}" x="{x0 + 0.5:.1f}" '
                     f'y="{f.y(c):.1f}" width="{max(0.5, x1 - x0 - 1):.1f}" '
                     f'height="{f.y(0) - f.y(c):.1f}"/>')
    for t in nice_ticks(edges[0], edges[-1], 6):
        if edges[0] <= t <= edges[-1]:
            parts.append(f'<text class="tick" x="{f.x(t):.1f}" y="{H - PAD_B + 18}" '
                         f'text-anchor="middle">{_fmt(t)}</text>')
    ox = f.x(observed)
    parts.append(f'<line class="observed" x1="{ox:.1f}" x2="{ox:.1f}" y1="{PAD_T}" '
                 f'y2="{H - PAD_B}"/><text class="legend" x="{ox + 6:.1f}" y="{PAD_T + 12}">'
                 f'observed {_fmt(observed)}</text>')
    parts.append(f'<text class="axis-label" x="{(W + PAD_L) / 2}" y="{H - 6}" '
                 f'text-anchor="middle">{escape(label)}</text>')
    return _svg("".join(parts), f"null distribution: {label}")
