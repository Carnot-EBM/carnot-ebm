#!/usr/bin/env python3
"""Live-agent scoreboard: levels solved, efficiency, and the 24h delta.

Operator asked for this at the end of every 30-minute health check: "a simple
scoreboard with game levels solved and an efficiency score with an improvement
score over the past 24 hours".

WHAT IT READS. Row files written by scripts/arc_scored_path_lever_harness.py --
the SCORED path (E3AgentPolicy), not the offline dev twin. The registry's 183
levels come from 25 hand-built per-game adapters and are NOT this number; the
scored agent has no adapters, which is the whole point of measuring it.

THE FIELDS ARE THE HARNESS'S OWN, not invented here:
  levels                       -- int per game. NOT levels_completed, a key that
                                  does not exist and which .get() turns into a
                                  false zero (that error cost a day of wrong
                                  reporting on 2026-08-21).
  efficiency_gateway_charged   -- the charged efficiency the harness computes.
                                  Preferred over `efficiency` because it is the
                                  gateway-charged figure rather than the
                                  optimistic local one.
  llm_on_row_valid             -- whether the row is a valid LLM-on measurement.
                                  A run can look healthy and be entirely invalid;
                                  three A/Bs in a row were. Invalid rows are
                                  counted and reported, never silently averaged.

COMPARING RUNS IS THE HARD PART, so it is done explicitly rather than by
subtracting headline numbers. Two runs may cover different game sets and
different generator configurations. Subtracting 16 levels over 25 games from 4
levels over 5 games says nothing. So the delta is computed over the INTERSECTION
of the two runs' game sets, and the intersection size is always printed. A delta
over fewer than 3 shared games is reported but marked thin.

CONFIG IS NOT IN THE ROWS. Whether the generator ran with thinking enabled is
not recorded per row, and it changes results substantially. This script cannot
read it and does not guess; it prints a reminder that the caller must state it.

Read-only. Touches no tracked file.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

# Row files live in job scratch, not in the repo: they are measurements in
# flight, not the research record. Override with --roots for another location.
DEFAULT_ROOTS = (
    Path(os.environ.get("CLAUDE_JOB_DIR", "/home/ianblenke/.claude/jobs/ad0c053d")) / "tmp",
)


def _unwrap(value):
    """Read through a principle-annotated field ({"value": ..., "principle": ...})."""
    return value.get("value") if isinstance(value, dict) and "value" in value else value


def load_rows(path: Path) -> list[dict]:
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    rows = data.get("rows") if isinstance(data, dict) else data
    if rows is None and isinstance(data, dict):
        rows = [data]
    return [r for r in (rows or []) if isinstance(r, dict)]


def summarize(rows: list[dict]) -> dict:
    """Aggregate one run. Absent is not zero: a missing `levels` is reported."""
    games = [r.get("game") for r in rows if r.get("game")]
    levels = 0
    missing_levels = 0
    for r in rows:
        v = _unwrap(r.get("levels"))
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            levels += int(v)
        else:
            missing_levels += 1
    effs = [
        _unwrap(r.get("efficiency_gateway_charged"))
        for r in rows
        if isinstance(_unwrap(r.get("efficiency_gateway_charged")), (int, float))
    ]
    valid = sum(1 for r in rows if _unwrap(r.get("llm_on_row_valid")) is True)
    actions = sum(
        int(_unwrap(r.get("actions")) or 0)
        for r in rows
        if isinstance(_unwrap(r.get("actions")), (int, float))
    )
    return {
        "games": len(rows),
        "game_ids": sorted(g for g in games if g),
        "levels": levels,
        "missing_levels": missing_levels,
        "valid": valid,
        "efficiency": (sum(effs) / len(effs)) if effs else None,
        "efficiency_n": len(effs),
        "actions": actions,
        "per_game": {r.get("game"): _unwrap(r.get("levels")) for r in rows if r.get("game")},
    }


def discover(roots) -> list[tuple[float, Path, list[dict]]]:
    """One entry per RUN, not per file.

    A single run is often split across several row files -- baseline25 wrote
    rows_first10.json and rows_rest15.json, and an A/B writes rows_off.json and
    rows_on.json. Treating each file as a run under-reports the headline: the
    first version of this script reported 9 levels over 15 games for a run that
    solved 16 over 25, because it picked the larger single FILE.

    So files are grouped by their parent directory and merged, deduplicated by
    game keeping the row from the NEWEST file. That also handles superseded
    files in the same directory -- baseline25 still holds
    rows_PRE_NOTHINK_partial.json from the invalid pre-fix attempt, and its
    stale ar25 row loses to the later one rather than corrupting the total.
    """

    # A/B ARMS ARE NOT A SEQUENCE. rows_off.json and rows_on.json in one
    # directory are two conditions of the same experiment, not an earlier and
    # later half of one run. Merging them newest-wins would silently report the
    # ON arm alone and hide the comparison. They are keyed separately; only
    # unsuffixed halves (rows.json, rows_first10.json, rows_rest15.json) merge.
    def _group_key(path: Path) -> Path:
        stem = path.stem
        for arm in ("_off", "_on"):
            if stem.endswith(arm):
                return path.parent / arm.lstrip("_")
        return path.parent

    by_dir: dict[Path, list[tuple[float, Path, list[dict]]]] = {}
    for root in roots:
        if not Path(root).is_dir():
            continue
        for p in Path(root).rglob("rows*.json"):
            rows = load_rows(p)
            if not rows or "levels" not in rows[0]:
                continue
            by_dir.setdefault(_group_key(p), []).append((p.stat().st_mtime, p, rows))

    runs: list[tuple[float, Path, list[dict]]] = []
    for parent, entries in by_dir.items():
        entries.sort(key=lambda t: t[0])  # oldest first, so newest overwrites
        merged: dict[str, dict] = {}
        for _, _, rows in entries:
            for r in rows:
                g = r.get("game")
                if g:
                    merged[g] = r
        newest_t, newest_p, _ = entries[-1]
        runs.append((newest_t, newest_p, list(merged.values())))
    runs.sort(key=lambda t: t[0], reverse=True)
    return runs


def overlap_delta(new_rows: list[dict], old_rows: list[dict]) -> dict | None:
    """Levels and efficiency delta over the games BOTH runs covered."""
    new_by = {r.get("game"): r for r in new_rows if r.get("game")}
    old_by = {r.get("game"): r for r in old_rows if r.get("game")}
    shared = sorted(set(new_by) & set(old_by))
    if not shared:
        return None
    n = summarize([new_by[g] for g in shared])
    o = summarize([old_by[g] for g in shared])
    return {
        "shared": shared,
        "new_levels": n["levels"],
        "old_levels": o["levels"],
        "new_eff": n["efficiency"],
        "old_eff": o["efficiency"],
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--roots", nargs="*", default=[str(r) for r in DEFAULT_ROOTS])
    ap.add_argument("--window-hours", type=float, default=24.0)
    args = ap.parse_args(argv)

    runs = discover(args.roots)
    if not runs:
        print("SCOREBOARD: no live-agent row files found under", args.roots)
        return 1

    # Latest = the most recent run with the most games, so a 3-game A/B in
    # flight does not displace a completed 25-game sweep as "current".
    newest_t = runs[0][0]
    recent = [r for r in runs if newest_t - r[0] < 6 * 3600]
    cur_t, cur_p, cur_rows = max(recent, key=lambda t: (len(t[2]), t[0]))
    cur = summarize(cur_rows)

    cutoff = time.time() - args.window_hours * 3600
    older = [r for r in runs if r[0] < cutoff]
    # Most RECENT run past the window, not the largest: the operator asked for
    # improvement "over the past 24 hours", so the baseline is where things
    # stood 24h ago. Picking the biggest older run instead silently compared
    # against a 4-day-old sweep.
    prev = max(older, key=lambda t: t[0]) if older else None

    print("")
    print("  ARC LIVE-AGENT SCOREBOARD (scored path, no per-game adapters)")
    print("  " + "-" * 62)
    eff = f"{cur['efficiency']:.4f}" if cur["efficiency"] is not None else "n/a"
    print(f"  levels solved      {cur['levels']:>6}   across {cur['games']} games")
    print(f"  efficiency         {eff:>6}   (gateway-charged, mean of {cur['efficiency_n']} rows)")
    print(f"  valid rows         {cur['valid']:>6} / {cur['games']}")
    print(f"  actions spent      {cur['actions']:>6}")
    if cur["missing_levels"]:
        print(f"  ** {cur['missing_levels']} row(s) had NO levels field -- absent, not zero")
    print(
        f"  source             {Path(cur_p).parent.name}/{Path(cur_p).name}"
        f"  ({time.strftime('%m-%d %H:%MZ', time.gmtime(cur_t))})"
    )

    print("")
    if prev is None:
        print(f"  24h delta          no run older than {args.window_hours:.0f}h to compare against")
    else:
        p_t, p_p, p_rows = prev
        d = overlap_delta(cur_rows, p_rows)
        print(
            f"  vs {Path(p_p).parent.name}/{Path(p_p).name}"
            f" ({time.strftime('%m-%d %H:%MZ', time.gmtime(p_t))})"
        )
        if d is None:
            print("  24h delta          NO SHARED GAMES -- runs are not comparable")
        else:
            dl = d["new_levels"] - d["old_levels"]
            line = f"  24h delta          {dl:+d} levels over {len(d['shared'])} shared game(s)"
            if len(d["shared"]) < 3:
                line += "   [thin: <3 games]"
            print(line)
            if d["new_eff"] is not None and d["old_eff"] is not None:
                print(
                    f"                     {d['new_eff'] - d['old_eff']:+.4f} efficiency"
                    f"   ({d['old_eff']:.4f} -> {d['new_eff']:.4f})"
                )
            print(f"                     shared: {', '.join(d['shared'])}")
    print("")
    print("  NOTE: generator config (thinking on/off) is not recorded per row and")
    print("        changes results substantially -- state it alongside these numbers.")
    print("")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
