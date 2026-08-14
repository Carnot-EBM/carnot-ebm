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
import ast
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


def _agentic_imports(path: Path) -> set[str]:
    """Sibling `carnot.agentic` modules this file imports, including function-level imports.

    Function-level imports matter here and an `ast`-only walk would still find them, but the
    regex is kept as a belt-and-braces second pass: `arc_loop_solve`-style code imports inside
    functions constantly, and a closure that misses one silently understates reachability -- which
    is the exact error direction this whole function exists to prevent.
    """
    try:
        src = path.read_text()
        tree = ast.parse(src)
    except (OSError, SyntaxError):
        return set()
    out: set[str] = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.ImportFrom) and n.module and "carnot.agentic." in n.module:
            out.add(n.module.split(".")[-1])
        elif isinstance(n, ast.Import):
            for a in n.names:
                if "carnot.agentic." in a.name:
                    out.add(a.name.split(".")[-1])
    out.update(re.findall(r"from carnot\.agentic\.(\w+) import", src))
    return out


# Which module each benchmark engine actually enters. Reachability is computed from here, so
# adding an engine to arc_bench without adding its entry point here would silently under-report
# what that engine can measure -- and under-reporting reachability is the direction that files
# real capabilities as duds.
ENGINE_ENTRY = {"explore": "arc_graph_explore", "scored": "arc_competition_agent"}


def reachable_flags(entry: str = "arc_graph_explore", engine: str | None = None) -> set[str]:
    """Flags the BENCHMARK can actually influence, by transitive import closure.

    WHY THIS GUARDS EVERY MEASUREMENT. `arc_bench.py` drives `graph_explore_solve_v2`, not the
    full `E3AgentPolicy` cascade. Measured 2026-08-13: 48 of the 95 tracked flags are inside that
    closure and 47 are not.

    Setting an unreachable flag and running the benchmark produces an identical result, because
    the code that reads it never executes. The comparison would then record HOLD -- "no level
    gained and no clear efficiency gain" -- which reads as "this capability is worthless". It
    would be wrong for 47 flags, and it would be wrong in the most damaging direction available:
    a ledger that systematically files real capabilities as duds, with evidence attached.

    So `--measure` refuses an unreachable flag instead of producing that number. Refusing to
    measure is a smaller error than measuring the wrong thing confidently.
    """
    if engine:
        entry = ENGINE_ENTRY.get(engine, entry)
    agentic = REPO / "python" / "carnot" / "agentic"
    seen: set[str] = set()
    frontier = {entry}
    while frontier:
        mod = frontier.pop()
        if mod in seen:
            continue
        seen.add(mod)
        p = agentic / f"{mod}.py"
        if p.exists():
            frontier |= _agentic_imports(p) - seen
    flags: set[str] = set()
    for mod in seen:
        p = agentic / f"{mod}.py"
        if p.exists():
            try:
                flags.update(re.findall(r"CARNOT_ARC_[A-Z0-9_]+", p.read_text()))
            except OSError:
                continue
    return flags


def _defining_modules(flag: str) -> list[str]:
    """Which agentic modules mention a flag. Used only to make an unreachable flag legible.

    Deliberately NOT an auto-classifier. Deciding "this is infrastructure, ignore it" from a module
    name is exactly the pattern list that drifts, and it would let a real capability disappear from
    the ledger because it happened to live in a file with the wrong name. Print the evidence and
    let a human classify.
    """
    out = []
    for d in SCAN_DIRS:
        for f in sorted(d.rglob("*.py")):
            try:
                if flag in f.read_text():
                    out.append(f.stem)
            except OSError:
                continue
    return out[:3]


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


def run_bench(
    env_overrides: dict[str, str] | None = None,
    out: Path | None = None,
    engine: str = "explore",
) -> dict:
    """Full sweep, as a subprocess, so a flag is applied the way the agent really reads it.

    In-process `os.environ` mutation would be faster and would not survive module-level constants
    that read their env var at import. A subprocess is the honest way to A/B an env flag.
    """
    out = out or Path(os.environ.get("TMPDIR", "/tmp")) / "arc_bench_arm.json"
    env = {**os.environ, **(env_overrides or {})}
    subprocess.run(
        [sys.executable, str(BENCH), "--all", "--quiet", "--engine", engine, "--out", str(out)],
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
    # Byte-identical across EVERY compared game. Computed here rather than inferred from the
    # aggregate, because two games could trade a level and leave the totals equal while the flag
    # was plainly doing something.
    identical = bool(set(b) & set(a)) and all(
        b[g]["levels_cleared"] == a[g]["levels_cleared"]
        and b[g]["actions_spent"] == a[g]["actions_spent"]
        for g in (set(b) & set(a))
        if not (b[g].get("error") or a[g].get("error"))
    )
    return {
        "games_compared": len(set(b) & set(a)),
        "identical_to_baseline": identical,
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
    # NO OBSERVABLE EFFECT comes FIRST, because it is not a weak result -- it is a different
    # claim, and conflating the two is how a real capability gets filed as a dud.
    #
    # Found on the first scored-engine measurement. CARNOT_ARC_HAZARD_MOVE_PRUNER produced
    # baseline and arm sweeps that were identical to the digit: 16 levels and 48,313 actions on
    # both. Not "a small effect" -- literally none. The flag is IMPORT-reachable on that engine,
    # so the reachability guard passed it, but it never FIRES: two pre-wiring censuses recorded
    # in arc_scored_path_lever_harness.py found this pruner fits 0 of 25 and 1 of 15 public games.
    #
    # "HOLD: no level gained" would read as "measured, does not help." The truth is "not measured
    # at all." Import reachability proves the code is LOADED; it cannot prove the code RUNS.
    if cmp.get("identical_to_baseline"):
        return False, (
            "UNINTERPRETABLE_NO_EFFECT: every game returned byte-identical levels AND actions. "
            "The flag changed nothing observable, so this is not evidence about its value -- it "
            "is evidence the code never fired, or is not wired to this engine. Do NOT record it "
            "as a null. Check the lever's own fire counters (see "
            "arc_scored_path_lever_harness.py) before concluding anything."
        )
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


def cmd_measure(flag: str, value: str, force: bool = False, engine: str = "explore") -> int:
    if not force and flag not in reachable_flags(engine=engine):
        print(
            f"arc-flag-ledger: REFUSING to measure {flag}.\n"
            f"  It is outside the {engine} engine's import closure, so the code that reads it\n"
            "  never runs. The sweep would return an identical result and this ledger would record\n"
            "  HOLD -- filing a real capability as worthless, with evidence attached.\n"
            + (
                "  It IS reachable on the `scored` engine -- re-run with --engine scored.\n"
                if flag in reachable_flags(engine="scored") and engine != "scored"
                else "  No engine reaches it. Check whether it is a tooling knob rather than an\n"
                "  agent capability (--status names the module that defines it).\n"
            )
            + "  Or pass --force to record the null deliberately."
        )
        return 1
    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    print(f"arc-flag-ledger: baseline sweep (flag unset)...")
    base = run_bench(out=tmp / "arc_bench_base.json", engine=engine)
    print(
        f"  baseline: {base['total_levels_cleared']} level(s), {base['total_actions_spent']} actions"
    )
    print(f"arc-flag-ledger: arm sweep ({flag}={value})...")
    arm = run_bench({flag: value}, out=tmp / "arc_bench_arm.json", engine=engine)
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
    entry["benchmark_reachable"] = flag in reachable_flags(engine=engine)
    entry["measured_on_engine"] = engine
    save(data)
    # Never let a cosmetic path-prettify raise. `relative_to` throws when the ledger is not under
    # the repo (a test tmp_path, an operator running with LEDGER overridden), and an exception
    # here would blow up AFTER an eleven-minute measurement has already been saved -- turning a
    # completed run into a traceback for the sake of a shorter filename.
    try:
        shown = LEDGER.relative_to(REPO)
    except ValueError:
        shown = LEDGER
    print(f"  recorded in {shown}")
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
    print("\n  reachable per engine:")
    covered: set[str] = set()
    for eng in ENGINE_ENTRY:
        r = set(flags) & reachable_flags(engine=eng)
        covered |= r
        print(f"    {eng:8} {len(r):>3} of {len(flags)}")
    missing = sorted(set(flags) - covered)
    if missing:
        print(
            f"    NEITHER  {len(missing):>3} -- no engine can see these. Naming the module that\n"
            "             defines each one, because 'N flags nobody can measure' reads as a\n"
            "             capability gap, and a tooling knob is not one:"
        )
        for f in missing:
            print(f"             {f:42} {_defining_modules(f)}")
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
    ap.add_argument(
        "--engine",
        choices=sorted(ENGINE_ENTRY),
        default="explore",
        help="which benchmark engine to measure on. `scored` runs E3AgentPolicy -- the path that "
        "actually ships -- and reaches 89 of 95 flags against `explore`'s 48, at roughly four "
        "times the wall clock.",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="measure a flag the benchmark cannot reach, recording the null deliberately",
    )
    ap.add_argument("--promote", metavar="FLAG")
    args = ap.parse_args(argv)

    if args.discover:
        return cmd_discover()
    if args.measure:
        return cmd_measure(args.measure, args.value, force=args.force, engine=args.engine)
    if args.promote:
        return cmd_promote(args.promote)
    return cmd_status()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
