#!/usr/bin/env python3
"""Parse ops/experiment-log.md and generate a static HTML dashboard at ops/dashboard.html.

The dashboard shows:
  - Experiment timeline with result indicators
  - Accuracy comparison bar chart (SVG)
  - Approach comparison table

No external dependencies required -- uses only the Python standard library.
Inline CSS/JS, SVG bars for charts.
"""

from __future__ import annotations

import html
import re
import sys
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Experiment:
    number: int
    name: str
    result_icon: str  # one of: pass, warn, fail
    result_text: str
    key_learning: str
    # Parsed numeric accuracy delta when present (e.g. +20, -12, 0)
    delta: float | None = None


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

# Matches summary-table rows like:
#   | 1 | SAT verify-and-repair (random) | check +91.6% energy reduction | ...
# Also matches row 14 which is outside the table block.
_ROW_RE = re.compile(
    r"\|\s*(\d+)\s*\|"           # experiment number
    r"\s*(.+?)\s*\|"             # experiment name
    r"\s*([^\|]+?)\s*\|"         # result column (icon + text)
    r"\s*(.+?)\s*\|",            # key learning
)

# Matches delta patterns like "+20%", "-12%", "+10%", "net zero", "+91.6%"
_DELTA_RE = re.compile(r"([+-]?\d+(?:\.\d+)?)\s*%")
_NET_ZERO_RE = re.compile(r"net zero", re.IGNORECASE)


def _parse_icon(raw: str) -> str:
    """Return 'pass', 'warn', or 'fail' based on the emoji in the result."""
    if "\u2705" in raw or "✅" in raw:
        return "pass"
    if "\u26a0" in raw or "⚠" in raw:
        return "warn"
    if "\u274c" in raw or "❌" in raw:
        return "fail"
    return "warn"


def _parse_delta(raw: str) -> float | None:
    """Extract numeric accuracy delta from result text."""
    if _NET_ZERO_RE.search(raw):
        return 0.0
    m = _DELTA_RE.search(raw)
    if m:
        return float(m.group(1))
    return None


def parse_experiment_log(path: Path) -> list[Experiment]:
    """Read the experiment log markdown and return structured experiment data."""
    text = path.read_text(encoding="utf-8")
    experiments: list[Experiment] = []
    for m in _ROW_RE.finditer(text):
        num = int(m.group(1))
        name = m.group(2).strip()
        result_raw = m.group(3).strip()
        learning = m.group(4).strip()
        icon = _parse_icon(result_raw)
        # Strip the emoji character(s) for clean display text
        result_text = re.sub(r"[✅⚠️❌]\s*", "", result_raw).strip()
        delta = _parse_delta(result_raw)
        experiments.append(Experiment(num, name, icon, result_text, learning, delta))
    return experiments


# ---------------------------------------------------------------------------
# HTML generation
# ---------------------------------------------------------------------------

_ICON_MAP = {
    "pass": ("&#x2705;", "#22c55e"),  # green check
    "warn": ("&#x26A0;&#xFE0F;", "#eab308"),  # yellow warning
    "fail": ("&#x274C;", "#ef4444"),  # red cross
}


def _bar_color(icon: str) -> str:
    return _ICON_MAP.get(icon, _ICON_MAP["warn"])[1]


def _svg_bar_chart(experiments: list[Experiment], width: int = 700, bar_h: int = 28, gap: int = 6) -> str:
    """Generate an SVG bar chart of accuracy deltas."""
    # Only include experiments that have a numeric delta
    items = [(e.number, e.name, e.delta, e.result_icon) for e in experiments if e.delta is not None]
    if not items:
        return "<p>No numeric accuracy deltas to chart.</p>"

    max_abs = max(abs(d) for _, _, d, _ in items) or 1
    chart_h = len(items) * (bar_h + gap) + 20
    mid_x = width // 2
    scale = (mid_x - 80) / max_abs  # leave room for labels

    lines: list[str] = []
    lines.append(f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{chart_h}" '
                 f'style="font-family:monospace;font-size:13px;">')
    # Zero line
    lines.append(f'<line x1="{mid_x}" y1="0" x2="{mid_x}" y2="{chart_h}" '
                 f'stroke="#666" stroke-dasharray="4"/>')

    for i, (num, name, delta, icon) in enumerate(items):
        y = i * (bar_h + gap) + 10
        bar_w = abs(delta) * scale
        color = _bar_color(icon)
        if delta >= 0:
            x = mid_x
        else:
            x = mid_x - bar_w

        # Bar
        lines.append(f'<rect x="{x:.1f}" y="{y}" width="{bar_w:.1f}" height="{bar_h}" '
                     f'fill="{color}" opacity="0.8" rx="3"/>')
        # Label left of bar: experiment number + short name
        short = html.escape(name[:30])
        lines.append(f'<text x="{mid_x - 5}" y="{y + bar_h - 8}" text-anchor="end" '
                     f'fill="#ccc" font-size="11">#{num} {short}</text>')
        # Value label
        sign = "+" if delta > 0 else ""
        val_x = (mid_x + bar_w + 5) if delta >= 0 else (mid_x - bar_w - 5)
        anchor = "start" if delta >= 0 else "end"
        lines.append(f'<text x="{val_x:.1f}" y="{y + bar_h - 8}" text-anchor="{anchor}" '
                     f'fill="#eee">{sign}{delta:.0f}%</text>')

    lines.append("</svg>")
    return "\n".join(lines)


def _timeline_html(experiments: list[Experiment]) -> str:
    """Generate a horizontal timeline of experiment result indicators."""
    dots: list[str] = []
    for e in experiments:
        icon_html, color = _ICON_MAP.get(e.result_icon, _ICON_MAP["warn"])
        tooltip = html.escape(f"#{e.number}: {e.name} — {e.result_text}")
        dots.append(
            f'<span class="dot" style="background:{color};" title="{tooltip}">'
            f'{e.number}</span>'
        )
    return '<div class="timeline">' + "".join(dots) + "</div>"


def _table_html(experiments: list[Experiment]) -> str:
    """Generate an HTML comparison table."""
    rows: list[str] = []
    for e in experiments:
        icon_html = _ICON_MAP.get(e.result_icon, _ICON_MAP["warn"])[0]
        delta_str = ""
        if e.delta is not None:
            sign = "+" if e.delta > 0 else ""
            delta_str = f"{sign}{e.delta:.0f}%"
        rows.append(
            f"<tr>"
            f"<td>{e.number}</td>"
            f"<td>{icon_html}</td>"
            f"<td>{html.escape(e.name)}</td>"
            f"<td class=\"mono\">{delta_str}</td>"
            f"<td>{html.escape(e.result_text)}</td>"
            f"<td>{html.escape(e.key_learning)}</td>"
            f"</tr>"
        )
    return "\n".join(rows)


def generate_html(experiments: list[Experiment]) -> str:
    """Build the full dashboard HTML string."""
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>Carnot Experiment Dashboard</title>
<style>
  :root {{
    --bg: #0f172a; --surface: #1e293b; --border: #334155;
    --text: #e2e8f0; --muted: #94a3b8;
    --green: #22c55e; --yellow: #eab308; --red: #ef4444;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
    background: var(--bg); color: var(--text);
    padding: 2rem; max-width: 1000px; margin: 0 auto;
  }}
  h1 {{ font-size: 1.6rem; margin-bottom: 0.5rem; }}
  h2 {{ font-size: 1.2rem; margin: 2rem 0 1rem; color: var(--muted); }}
  .subtitle {{ color: var(--muted); margin-bottom: 2rem; }}
  .card {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1.5rem; margin-bottom: 1.5rem;
  }}
  /* Timeline */
  .timeline {{ display: flex; gap: 8px; flex-wrap: wrap; }}
  .dot {{
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 12px; font-weight: 700; color: #000;
    cursor: default;
  }}
  /* Table */
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ text-align: left; padding: 0.6rem; border-bottom: 2px solid var(--border); color: var(--muted); }}
  td {{ padding: 0.6rem; border-bottom: 1px solid var(--border); }}
  tr:hover {{ background: rgba(255,255,255,0.03); }}
  .mono {{ font-family: monospace; }}
  /* Chart container */
  .chart {{ overflow-x: auto; }}
  /* Stats row */
  .stats {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
  .stat {{
    background: var(--surface); border: 1px solid var(--border);
    border-radius: 8px; padding: 1rem 1.5rem; flex: 1; min-width: 140px; text-align: center;
  }}
  .stat-value {{ font-size: 1.8rem; font-weight: 700; }}
  .stat-label {{ font-size: 0.8rem; color: var(--muted); margin-top: 0.2rem; }}
  footer {{ margin-top: 3rem; text-align: center; color: var(--muted); font-size: 0.75rem; }}
</style>
</head>
<body>
<h1>Carnot Experiment Dashboard</h1>
<p class="subtitle">Tracking {len(experiments)} experiments across the EBM research pipeline</p>

<div class="stats">
  <div class="stat">
    <div class="stat-value" style="color:var(--green)">{sum(1 for e in experiments if e.result_icon == 'pass')}</div>
    <div class="stat-label">Passed</div>
  </div>
  <div class="stat">
    <div class="stat-value" style="color:var(--yellow)">{sum(1 for e in experiments if e.result_icon == 'warn')}</div>
    <div class="stat-label">Partial</div>
  </div>
  <div class="stat">
    <div class="stat-value" style="color:var(--red)">{sum(1 for e in experiments if e.result_icon == 'fail')}</div>
    <div class="stat-label">Failed</div>
  </div>
  <div class="stat">
    <div class="stat-value">{len(experiments)}</div>
    <div class="stat-label">Total</div>
  </div>
</div>

<h2>Experiment Timeline</h2>
<div class="card">
  {_timeline_html(experiments)}
</div>

<h2>Accuracy Delta (% change)</h2>
<div class="card chart">
  {_svg_bar_chart(experiments)}
</div>

<h2>Experiment Details</h2>
<div class="card" style="overflow-x:auto;">
  <table>
    <thead>
      <tr><th>#</th><th></th><th>Experiment</th><th>Delta</th><th>Result</th><th>Key Learning</th></tr>
    </thead>
    <tbody>
      {_table_html(experiments)}
    </tbody>
  </table>
</div>

<footer>Generated by scripts/generate_dashboard.py from ops/experiment-log.md</footer>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    log_path = repo_root / "ops" / "experiment-log.md"
    out_path = repo_root / "ops" / "dashboard.html"

    if not log_path.exists():
        print(f"ERROR: {log_path} not found", file=sys.stderr)
        sys.exit(1)

    experiments = parse_experiment_log(log_path)
    if not experiments:
        print("WARNING: No experiments parsed from log", file=sys.stderr)

    html_content = generate_html(experiments)
    out_path.write_text(html_content, encoding="utf-8")
    print(f"Dashboard generated: {out_path}")
    print(f"  {len(experiments)} experiments parsed")
    passed = sum(1 for e in experiments if e.result_icon == "pass")
    print(f"  {passed} passed, "
          f"{sum(1 for e in experiments if e.result_icon == 'warn')} partial, "
          f"{sum(1 for e in experiments if e.result_icon == 'fail')} failed")


if __name__ == "__main__":
    main()
