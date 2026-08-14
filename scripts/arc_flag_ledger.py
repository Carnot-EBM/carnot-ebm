#!/usr/bin/env python3
"""Turn measured ARC improvements ON by themselves, and record why.

WHY THIS EXISTS
---------------
Measured 2026-08-13: the agent carries **101 distinct `CARNOT_ARC_*` flags**, and over the last 10
milestones 13 of 16 ARC tasks ended in `ready_no_solve_claim` or `default_off`. Every one shipped a
capability behind a flag and turned nothing on. Nothing anywhere records which flags are on, why,
or on what evidence.

That is a loop with generation and no selection. It accumulates options instead of improving, and
101 unchosen options is worse than 10 -- it is a search space nobody searches.

This is the selection layer. A flag that beats baseline on `scripts/arc_bench.py` gets promoted to
default-on, automatically, with the evidence written down and the promotion reversible.

WHY "k INDEPENDENT RUNS" MEANS k DIFFERENT GAMES, NOT k REPEATS
----------------------------------------------------------------
`graph_explore_solve_v2` is deterministic. Measured directly: dc22 spent exactly 11,737 actions on
two separate runs. So repeating the benchmark is not replication -- it is the same computation
twice, and averaging it would manufacture false confidence from a sample of one.

Independence here comes from GAMES. A promotion needs the flag to help on a broad slice of the
roster, which is why `--measure` runs the full sweep and the promotion rule counts games improved
against games regressed. Anyone adding a stochastic component to the search must revisit this
paragraph; the rule assumes determinism and would be wrong without it.

THE REGRESSION GATE (and why it is not optional)
--------------------------------------------------
Turning things on unsupervised is how working behaviour dies. The precedent is on the record: the
ARC engine store overwrote unconditionally and destroyed ka59 from 1.0 to 0.0, with retention worth
p=4.9e-5. So a promotion that improves the aggregate while breaking one game is REFUSED, and the
game that broke is named. An average is exactly the statistic that hides this.

WHAT IT NEVER DOES. It never edits agent source. Promotion writes a default into
`ops/arc_flag_ledger.yaml`, which the agent reads; the code keeps its own fallback. It never blocks
a commit. It never promotes a flag it has not measured.

Usage:
    python3 scripts/arc_flag_ledger.py --discover            # find flags, add as unevaluated
    python3 scripts/arc_flag_ledger.py --status
    python3 scripts/arc_flag_ledger.py --measure CARNOT_ARC_X --value 1
    python3 scripts/arc_flag_ledger.py --promote CARNOT_ARC_X
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parents[1]
LEDGER = REPO / "ops" / "arc_flag_ledger.yaml"
BENCH = REPO / "scripts" / "arc_bench.py"
SCAN_DIRS = (REPO / "python" / "carnot" / "agentic",)

# A flag must help on at least this many games and harm none. Deliberately strict on the harm
# side: see the regression paragraph above.
MIN_GAMES_IMPROVED = 2


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


def discover_flags() -> list[str]:
    """Every `CARNOT_ARC_*` name the agent mentions.

    Read from source rather than kept as a list here, for the same reason `arc_bench.roster` reads
    the registry: a hand-maintained list of what to watch drifts narrower than the thing it
    watches, which is this project's most-repeated defect.
    """
    found: set[str] = set()
    for d in SCAN_DIRS:
        for f in d.rglob("*.py"):
            try:
                found.update(re.findall(r"CARNOT_ARC_[A-Z0-9_]+", f.read_text()))
            except OSError:
                continue
    return sorted(found)


def load() -> dict[str, Any]:
    import yaml

    if not LEDGER.exists():
        return {"schema": "carnot.arc_flag_ledger.v1", "flags": {}}
    try:
        return yaml.safe_load(LEDGER.read_text()) or {
            "schema": "carnot.arc_flag_ledger.v1",
            "flags": {},
        }
    except Exception:  # noqa: BLE001
        # Fail loud rather than silently starting a fresh ledger over a corrupt one -- that would
        # erase every recorded promotion and its evidence.
        raise SystemExit(f"arc-flag-ledger: {LEDGER} is unreadable; fix or move it, do not ignore")


def save(data: dict) -> None:
    import yaml

    LEDGER.write_text(yaml.safe_dump(data, sort_keys=True, width=100))


def run_bench(env_overrides: dict[str, str] | None = None, out: Path | None = None) -> dict:
    """Full sweep, as a subprocess, so a flag is applied the way the agent really reads it.

    In-process `os.environ` mutation would be faster and would not survive module-level constants
    that read their env var at import. A subprocess is the honest way to A/B an env flag.
    """
    out = out or Path(os.environ.get("TMPDIR", "/tmp")) / "arc_bench_arm.json"
    env = {**os.environ, **(env_overrides or {})}
    subprocess.run(
        [sys.executable, str(BENCH), "--all", "--quiet", "--out", str(out)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        timeout=3600,
    )
    try:
        return json.loads(out.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"arc-flag-ledger: benchmark produced no readable report ({exc})")


def compare(baseline: dict, arm: dict) -> dict[str, Any]:
    """Per-game comparison. The aggregate is reported but never decides.

    Levels decide, actions break ties. A flag that clears the same levels using fewer actions is a
    real improvement -- the scored metric squares efficiency -- but it must never outweigh a lost
    level, so the two are kept apart rather than folded into one score.
    """
    b = {r["game"]: r for r in baseline.get("per_game_rows", [])}
    a = {r["game"]: r for r in arm.get("per_game_rows", [])}
    improved, regressed, cheaper, costlier = [], [], [], []
    for g in sorted(set(b) & set(a)):
        if b[g].get("error") or a[g].get("error"):
            continue
        db = a[g]["levels_cleared"] - b[g]["levels_cleared"]
        if db > 0:
            improved.append(g)
        elif db < 0:
            regressed.append(g)
        elif a[g]["levels_cleared"] > 0:
            da = a[g]["actions_spent"] - b[g]["actions_spent"]
            (cheaper if da < 0 else costlier if da > 0 else []).append(g)  # type: ignore[union-attr]
    return {
        "games_compared": len(set(b) & set(a)),
        "levels_improved": improved,
        "levels_regressed": regressed,
        "same_levels_cheaper": cheaper,
        "same_levels_costlier": costlier,
        "baseline_total_levels": baseline.get("total_levels_cleared"),
        "arm_total_levels": arm.get("total_levels_cleared"),
    }


def verdict(cmp: dict) -> tuple[bool, str]:
    """Promote or not, and say why in one sentence a human can check.

    Refusing on ANY regression is the point. An aggregate improvement that costs one game is the
    ka59 failure repeating with better paperwork.
    """
    if cmp["levels_regressed"]:
        return False, (
            f"REFUSED: regressed on {cmp['levels_regressed']}. A flag that breaks a game it used "
            "to clear is not promoted, whatever the aggregate says."
        )
    n = len(cmp["levels_improved"])
    if n >= MIN_GAMES_IMPROVED:
        return True, f"PROMOTE: cleared new levels on {cmp['levels_improved']}, regressed none."
    if n:
        return False, (
            f"HOLD: improved only {cmp['levels_improved']} ({n} game, needs "
            f"{MIN_GAMES_IMPROVED}). One game is a coincidence, not a capability."
        )
    if cmp["same_levels_cheaper"] and not cmp["same_levels_costlier"]:
        return True, (
            f"PROMOTE: same levels, fewer actions on {cmp['same_levels_cheaper']}, costlier on "
            "none. The scored metric squares efficiency, so this is a real gain."
        )
    return False, "HOLD: no level gained and no clear efficiency gain."


def cmd_measure(flag: str, value: str) -> int:
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    print(f"arc-flag-ledger: baseline sweep (flag unset)...")
    base = run_bench(out=tmp / "arc_bench_base.json")
    print(
        f"  baseline: {base['total_levels_cleared']} level(s), {base['total_actions_spent']} actions"
    )
    print(f"arc-flag-ledger: arm sweep ({flag}={value})...")
    arm = run_bench({flag: value}, out=tmp / "arc_bench_arm.json")
    print(
        f"  arm:      {arm['total_levels_cleared']} level(s), {arm['total_actions_spent']} actions"
    )

    cmp = compare(base, arm)
    ok, why = verdict(cmp)
    print(f"\n  {why}")

    data = load()
    entry = data.setdefault("flags", {}).setdefault(flag, {"state": "unevaluated", "evidence": []})
    entry["evidence"].append({"date": _now(), "value": value, "verdict": why, **cmp})
    entry["last_measured"] = _now()
    entry["promotable"] = ok
    save(data)
    print(f"  recorded in {LEDGER.relative_to(REPO)}")
    return 0


def cmd_promote(flag: str) -> int:
    """Promote only on evidence already recorded. Never measures implicitly.

    A promote that quietly runs its own measurement is a promote nobody reviewed. Measuring is a
    separate, visible step.
    """
    data = load()
    entry = (data.get("flags") or {}).get(flag)
    if not entry:
        print(f"arc-flag-ledger: {flag} has no ledger entry. Run --measure first.")
        return 1
    if not entry.get("promotable"):
        last = (entry.get("evidence") or [{}])[-1].get("verdict", "never measured")
        print(f"arc-flag-ledger: REFUSING to promote {flag}.\n  {last}")
        return 1
    entry["state"] = "on"
    entry["promoted_on"] = _now()
    save(data)
    print(f"arc-flag-ledger: promoted {flag} -> default ON, evidence recorded.")
    return 0


def cmd_discover() -> int:
    data = load()
    flags = data.setdefault("flags", {})
    added = 0
    for f in discover_flags():
        if f not in flags:
            flags[f] = {"state": "unevaluated", "discovered": _now(), "evidence": []}
            added += 1
    save(data)
    print(f"arc-flag-ledger: {len(flags)} flag(s) tracked, {added} newly discovered.")
    return 0


def cmd_status() -> int:
    data = load()
    flags = data.get("flags") or {}
    by = {}
    for name, e in flags.items():
        by.setdefault(e.get("state", "unevaluated"), []).append(name)
    print(f"arc-flag-ledger: {len(flags)} flag(s) tracked")
    for state in sorted(by):
        print(f"  {state}: {len(by[state])}")
    for name in sorted(by.get("on", [])):
        e = flags[name]
        print(f"    ON  {name}  (promoted {e.get('promoted_on')})")
    unevaluated = len(by.get("unevaluated", []))
    if unevaluated:
        print(
            f"\n  {unevaluated} flag(s) have never been measured. An unmeasured flag is an option\n"
            "  nobody can choose between -- that is the condition this ledger exists to end."
        )
    return 0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--measure", metavar="FLAG")
    ap.add_argument("--value", default="1")
    ap.add_argument("--promote", metavar="FLAG")
    args = ap.parse_args(argv)

    if args.discover:
        return cmd_discover()
    if args.measure:
        return cmd_measure(args.measure, args.value)
    if args.promote:
        return cmd_promote(args.promote)
    return cmd_status()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
