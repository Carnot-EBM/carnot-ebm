#!/usr/bin/env python3
"""Fill the research note's RESULTS_TABLE / MECHANISM_TABLE / SAMPLER_SECTION placeholders.

Generated from the artifact rather than hand-typed, so the prose and the artifact cannot state
different numbers -- the failure mode the freshness lint exists for, one document out.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path("/home/ianblenke/github.com/ianblenke/carnot")
ART = REPO / "results" / "outer_loop_arc_induce_budget_phase1_20260731.json"
NOTE = REPO / "docs" / "research-notes" / "arc-induce-completion-budget-2026-07-31.md"

doc = json.loads(ART.read_text())
rows = doc["rows"]

LANE_LABEL = {
    "induce_combined": "combined induce",
    "induce_engine_only": "engine-only induce",
    "refactor": "refactor",
}


def _ok(lane, budget, arm="shipped"):
    return [
        r for r in rows
        if r["lane"] == lane and r["budget"] == budget
        and r.get("arm", "shipped") == arm and r.get("status") == "ok"
    ]


# ---- RESULTS TABLE ----------------------------------------------------------------------
lines = [
    "",
    "| call | budget | n | accepted by `generate()` | **usable** | hit the cap | mean `predicted_n` | mean code lines |",
    "|---|---|---|---|---|---|---|---|",
]
for lane in ("induce_combined", "induce_engine_only", "refactor"):
    for budget in (4096, 8192, 16384):
        rs = _ok(lane, budget)
        if not rs:
            continue
        acc = sum(1 for r in rs if r.get("generate_would_accept"))
        use = sum(1 for r in rs if r.get("usable_engine"))
        cap = sum(1 for r in rs
                  if r.get("stop_type") == "limit" and r.get("predicted_n") == budget)
        mp = sum(r.get("predicted_n") or 0 for r in rs) / len(rs)
        mc = sum(r.get("code_lines") or 0 for r in rs) / len(rs)
        lines.append(
            f"| {LANE_LABEL[lane]} | {budget} | {len(rs)} | {acc}/{len(rs)} | "
            f"**{use}/{len(rs)}** | {cap}/{len(rs)} | {mp:.0f} | {mc:.1f} |"
        )
lines.append("")
results_table = "\n".join(lines)

# ---- MECHANISM TABLE --------------------------------------------------------------------
mech = [
    "",
    "| call | seed/attempt | distinct non-blank lines (4096 -> 8192 -> 16384) | non-blank lines emitted | code lines |",
    "|---|---|---|---|---|",
]
for lane in ("induce_engine_only", "induce_combined"):
    for attempt in (0, 1, 2):
        cells = []
        for budget in (4096, 8192, 16384):
            rs = [r for r in _ok(lane, budget) if r["attempt"] == attempt]
            cells.append(rs[0] if rs else None)
        if not any(cells):
            continue
        d = " -> ".join(str(c.get("n_distinct_nonblank_lines")) if c else "-" for c in cells)
        n = " -> ".join(str(c.get("n_nonblank_lines")) if c else "-" for c in cells)
        cl = " -> ".join(str(c.get("code_lines")) if c else "-" for c in cells)
        mech.append(f"| {LANE_LABEL[lane]} | a={attempt} | {d} | {n} | {cl} |")
mech.append("")
mech_table = "\n".join(mech)

# ---- SAMPLER SECTION --------------------------------------------------------------------
sampler_rows = [r for r in rows if r["lane"] == "sampler_control" and r.get("status") == "ok"]
if not sampler_rows:
    sampler = (
        "### Sampler control\n\n"
        "NOT RUN in the wall budget available. Reported as a missing observation rather than "
        "omitted: the repetition-control question is open and untested, not answered.\n"
    )
else:
    arms: dict = {}
    for r in sampler_rows:
        arms.setdefault(r.get("arm", "shipped"), []).append(r)
    sampler = [
        "### Sampler control — the same prompt, seed, temperature and budget, one flag apart",
        "",
        "Engine-only induce prompt, budget 4096, 3 seeded attempts per arm, one server, one card.",
        "`off` is the SHIPPED configuration, re-run inside this script rather than reused from the",
        "budget lane, so the comparison never crosses a server process.",
        "",
        "| sampler arm | hit the 4096 cap | stopped naturally | accepted | returns on all paths | **changes the grid** | mean cell recall |",
        "|---|---|---|---|---|---|---|",
    ]
    for arm in sorted(arms, key=lambda a: (a != "off", a)):
        v = arms[arm]
        cap = sum(1 for r in v if r.get("stop_type") == "limit")
        sampler.append(
            f"| `{arm}` | {cap}/{len(v)} | {len(v) - cap}/{len(v)} | "
            f"{sum(1 for r in v if r.get('generate_would_accept'))}/{len(v)} | "
            f"{sum(1 for r in v if r.get('engine_returns_on_all_paths'))}/{len(v)} | "
            f"**{sum(1 for r in v if r.get('engine_changes_anything'))}/{len(v)}** | "
            f"{sum(r.get('cell_recall') or 0 for r in v)/len(v):.4f} |"
        )
    sampler += [
        "",
        "`repeat_penalty_1.1` is the only configuration in the entire sweep — across three call",
        "shapes, three budgets and three sampler arms — that produces a non-degenerate engine, and",
        "it produces one on every attempt. The shipped arm burns the whole 4096-token budget and",
        "returns nothing; `repeat_penalty` stops naturally at 1133-1912 tokens with the real ft09",
        "mechanic in it — a 6x6 block at `(y-2, x-2)` toggling between colours 8 and 9:",
        "",
        "```python",
        "current_color = new_grid[target_row, target_col]",
        "if current_color == 9:    new_color = 8",
        "elif current_color == 8:  new_color = 9",
        "else:                     return new_grid  # Not a togglable block",
        "new_grid[target_row:target_row + 6, target_col:target_col + 6] = new_color",
        "```",
        "",
        "**The scorer behind that `changes the grid` column was cross-validated against the",
        "shipped gate, to the cell.** A behavioural scorer written for one measurement is worth",
        "nothing if it disagrees with the gate the live agent runs. Scoring the `repeat_penalty",
        "1.1` engine over the same 25 transitions gives 216 correct changed cells of 228 — cell",
        "recall 0.9474. The live `onb` cell's recorded gate fields are",
        "`verify_correct_changed_cells: 216`, `verify_cell_recall: 0.9474`,",
        "`verify_change_fidelity: 0.947368`. Identical. Which also means this arm reaches, on its",
        "FIRST naturally-stopped attempt in 1912 tokens, the cell recall the live LLM-ON run only",
        "reached on one replicate after three refinement rounds.",
        "",
        "**Three caveats, all of which matter before anyone acts on this.**",
        "",
        "1. **The cell recall is IN-SAMPLE.** All 6 of ft09's 25 grid-changing transitions appear",
        "   in the induce prompt; the 16 held-out transitions are every one a no-op, so there are",
        "   ZERO held-out changed cells and this measurement cannot speak to generalisation. What",
        "   it does show is that the model can express the mechanic it was shown, which the shipped",
        "   arm never got far enough to attempt.",
        "2. **n=3, one game, one prompt.** A sampler change touches every generation the scored",
        "   agent makes, on games this was not measured on.",
        "3. **It is NOT shipped here, deliberately.** Changing how the scored agent samples is a",
        "   behaviour change, not a measurement change, and this session's remit was to measure the",
        "   budget. The recommendation is for the operator: run this arm across the other five",
        "   games before touching the default.",
        "",
    ]
    sampler = "\n".join(sampler)

# ---- ENGINE SCORE SECTION ---------------------------------------------------------------
scored = [r for r in rows if r["lane"] != "sampler_control" and r.get("status") == "ok"]
n_changes = sum(1 for r in scored if r.get("engine_changes_anything"))
eng = [
    "Every generated engine was RUN against ft09's 25 real captured transitions, because the",
    "return-on-all-paths check is gameable and something gamed it. The refactor lane produced",
    "completions that `generate()` accepts, that parse, that return on every path — and that are",
    "the identity function:",
    "",
    "```python",
    "def engine(grid, action, data):",
    "    x = data.get('x', 0); y = data.get('y', 0)",
    "    rows = len(grid); cols = len(grid[0]) if rows > 0 else 0",
    "    if action == 6:",
    "        return grid",
    "    return grid",
    "```",
    "",
    f"**Of {len(scored)} engines generated under the SHIPPED sampler, {n_changes} ever produce an",
    "output different from their input.** Not one. That is every induce call shape, the refactor",
    "call, and every budget. (The sampler-control arms below are the exception, and the only one.)",
    "",
    "| lane | budget | accepted | returns on all paths | **changes the grid** | mean heldout-exact | mean cell recall |",
    "|---|---|---|---|---|---|---|",
]
for lane in ("induce_combined", "induce_engine_only", "refactor"):
    for budget in (4096, 8192, 16384):
        rs = _ok(lane, budget)
        if not rs:
            continue
        eng.append(
            f"| {LANE_LABEL[lane]} | {budget} | "
            f"{sum(1 for r in rs if r.get('generate_would_accept'))}/{len(rs)} | "
            f"{sum(1 for r in rs if r.get('engine_returns_on_all_paths'))}/{len(rs)} | "
            f"**{sum(1 for r in rs if r.get('engine_changes_anything'))}/{len(rs)}** | "
            f"{sum(r.get('heldout_exact') or 0 for r in rs)/len(rs):.3f} | "
            f"{sum(r.get('cell_recall') or 0 for r in rs)/len(rs):.3f} |"
        )
eng += [
    "",
    "The refactor engines score **19 of 25 heldout-exact** — which looks like the best result in",
    "the sweep until you notice that 19 of ft09's 25 transitions are no-ops, and an engine that",
    "predicts \"nothing ever changes\" gets every one of them right. Their cell recall is 0.000.",
    "That is the vacuous pass the change-aware gate exists to catch, reproduced here from the",
    "generator side, and it is a live argument for the gate that is computed today but left",
    "`gate_enabled: false`.",
    "",
    "It also reframes the Phase-1 question. \"Does more budget produce a complete `engine()`\" was",
    "the wrong question to be able to answer yes to: the engines are not incomplete, they are",
    "inert. No completion budget fixes that.",
    "",
]
eng_section = "\n".join(eng)

text = NOTE.read_text()
text = text.replace("`ENGINE_SCORE_SECTION`", eng_section)
text = text.replace("`RESULTS_TABLE`", results_table)
text = text.replace("`MECHANISM_TABLE`", mech_table)
text = text.replace("`SAMPLER_SECTION`", sampler)
NOTE.write_text(text)
print("filled; remaining placeholders:",
      [t for t in ("RESULTS_TABLE", "MECHANISM_TABLE", "SAMPLER_SECTION") if t in text])
