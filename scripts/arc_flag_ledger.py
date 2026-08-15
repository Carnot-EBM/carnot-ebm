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
from functools import lru_cache
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


@lru_cache(maxsize=1)
def discover_flags() -> tuple[str, ...]:
    """Every `CARNOT_ARC_*` name the agent mentions.

    Read from source rather than kept as a list here, for the same reason `arc_bench.roster` reads
    the registry: a hand-maintained list of what to watch drifts narrower than the thing it
    watches, which is this project's most-repeated defect.
    """
    found: set[str] = set()
    for _stem, src in _agent_sources():
        found.update(re.findall(r"CARNOT_ARC_[A-Z0-9_]+", src))
    return tuple(sorted(found))


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


@lru_cache(maxsize=1)
def _agent_sources() -> tuple[tuple[str, str], ...]:
    """(module stem, source) for every agentic file, read ONCE.

    Cached because the callers are per-flag and there are ~95 flags over ~200 files. Uncached,
    classifying the whole set re-read the tree ninety-five times -- roughly 19,000 file reads,
    slow enough that the sweep's own candidate listing looked like a hang. Correctness was never
    at stake; the cache just stops the quadratic.
    """
    out = []
    for d in SCAN_DIRS:
        for f in sorted(d.rglob("*.py")):
            try:
                out.append((f.stem, f.read_text()))
            except OSError:
                continue
    return tuple(out)


@lru_cache(maxsize=512)
def _read_sites(flag: str) -> tuple[tuple[str, str], ...]:
    """(text before, text after) each `environ.get("<flag>"...)` in the agent source."""
    out = []
    if True:
        for _stem, src in _agent_sources():
            for m in re.finditer(
                rf'([\w.\[\]()"\'= ]{{0,60}})environ\.get\(\s*["\']{flag}["\']([^\n]{{0,80}})', src
            ):
                out.append((m.group(1), m.group(2)))
    return tuple(out)


@lru_cache(maxsize=512)
def classify_flag(flag: str) -> str:
    """`bool` | `numeric` | `path` | `inverse` | `unknown`, from how the value is USED.

    WHY A SWEEP CANNOT JUST SET EVERYTHING TO "1". Of the 54 default-off flags the scored engine
    reaches, only 23 are boolean toggles. The rest would be actively corrupted by `=1`:
    `CARNOT_ARC_GGUF_PATH=1` is a nonsense model path, `CARNOT_ARC_INDUCE_TIMEOUT=1` is a
    one-second timeout that breaks induction outright, and `CARNOT_ARC_DISABLE_INDUCTION=1` turns
    the LLM OFF -- the exact opposite of measuring a capability.

    A blind sweep would run all three, watch the numbers fall, and record the damage as evidence
    that those capabilities are harmful. That is worse than not sweeping: it is a ledger full of
    confident, wrong verdicts, which is the failure mode every guard in this file exists to avoid.

    `inverse` is separate from `bool` because the name carries the semantics: turning on a
    `DISABLE_*` flag REMOVES a capability, so "does it improve the benchmark" is the wrong
    question and a regression would be the expected result.

    UNKNOWN IS NOT SWEPT. 33 flags land there, and some are plainly numeric on inspection
    (`GOAL_GATE_MAX_NODES`, `GENERATOR_SEED`). Under-sweeping costs coverage; over-sweeping costs
    correctness and fills the record with garbage. The error direction is deliberate.
    """
    if re.search(r"DISABLE|_OFF\b|_DISABLED", flag):
        return "inverse"
    # GUARD / PERMISSION flags are not capabilities. Turning one on removes a protection or
    # asserts a precondition; "does it improve the benchmark" is the wrong question and the honest
    # answer is always "no, that is not what it is for".
    #
    # `CARNOT_ARC_E3_ALLOW_EVIDENCE_WRITE` is why this branch exists. It disables the guard that
    # stops a test writing into the TRACKED e3 evidence directory. It is inert outside pytest, so
    # sweeping it would have produced a confident, meaningless null -- and the name sitting in a
    # candidate list called "capability flags" is a trap for the next reader, who would reasonably
    # conclude someone had measured it and found it worthless. Given the day this project has had
    # with tests rewriting the research record, a sweep must not be in the habit of flipping
    # write-permission flags at all.
    if re.search(r"ALLOW_|_ALLOW|REQUIRE_|_BYPASS|BYPASS_", flag):
        return "guard"
    kinds = set()
    for pre, post in _read_sites(flag):
        if re.search(r"int\s*\(|float\s*\(", pre):
            kinds.add("numeric")
        elif re.search(r'==\s*["\']1["\']|in\s*\(?["\']1["\']|\.lower\(\)\s*in', pre + post):
            kinds.add("bool")
        elif re.search(r"Path\s*\(", pre) or re.search(r"_DIR$|_PATH$", flag):
            kinds.add("path")
        else:
            kinds.add("unknown")
    if not kinds:
        return "unknown"
    for k in ("numeric", "path"):  # a value knob anywhere wins: never set it to "1"
        if k in kinds:
            return k
    return "bool" if kinds == {"bool"} else "unknown"


def _defining_modules(flag: str) -> list[str]:
    """Which agentic modules mention a flag. Used only to make an unreachable flag legible.

    Deliberately NOT an auto-classifier. Deciding "this is infrastructure, ignore it" from a module
    name is exactly the pattern list that drifts, and it would let a real capability disappear from
    the ledger because it happened to live in a file with the wrong name. Print the evidence and
    let a human classify.
    """
    return [stem for stem, src in _agent_sources() if flag in src][:3]


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
    # 4 hours, not 1. The first real sweep died at flag 16 of 20 on a TimeoutExpired: the scored
    # engine's 25-game sweep takes ~17 min at baseline, but a flag that slows the search can push
    # it far past that, and CARNOT_ARC_SGE_CANDIDATE_ROUTER pushed it past the hour. A slow arm is
    # a RESULT (the flag costs time), not a reason to lose the whole run.
    timeout_s = int(os.environ.get("CARNOT_BENCH_TIMEOUT_S", 4 * 3600))
    try:
        subprocess.run(
            [sys.executable, str(BENCH), "--all", "--quiet", "--engine", engine, "--out", str(out)],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        # Return a TIMED-OUT marker rather than raising. Raising killed the sweep and lost the
        # four flags after it; a timeout is information about THIS flag and must not end the run.
        # `timed_out` has no per_game_rows, so `compare` sees no games and `verdict` cannot read
        # it as an improvement -- it degrades to "no games compared", never to a false promotion.
        return {"timed_out": True, "timeout_s": timeout_s, "per_game_rows": []}
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
    # Did ANY lever's internal counters move, even though the outcome did not? This separates
    # "the flag never resolved" (a wiring bug) from "the flag ran and changed nothing" (a weak but
    # real null). Fully general -- it diffs whatever `*_diagnostics()` blocks the rows happen to
    # carry, with no per-lever knowledge, so a lever added tomorrow is covered.
    diag_moved = any(
        (b[g].get("fire_counters") or {}) != (a[g].get("fire_counters") or {})
        for g in (set(b) & set(a))
    )
    identical = bool(set(b) & set(a)) and all(
        b[g]["levels_cleared"] == a[g]["levels_cleared"]
        and b[g]["actions_spent"] == a[g]["actions_spent"]
        for g in (set(b) & set(a))
        if not (b[g].get("error") or a[g].get("error"))
    )
    return {
        "games_compared": len(set(b) & set(a)),
        "arm_timed_out": bool(arm.get("timed_out")),
        "identical_to_baseline": identical,
        "lever_counters_moved": diag_moved,
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
    if cmp.get("arm_timed_out"):
        return False, (
            "UNINTERPRETABLE_TIMED_OUT: the arm sweep exceeded its wall-clock cap, so no outcome "
            "was measured. That is a fact about cost, not about capability -- a flag that slows "
            "the search this much may still be correct. Re-run it alone with a longer "
            "CARNOT_BENCH_TIMEOUT_S before concluding anything."
        )
    if cmp.get("identical_to_baseline"):
        if cmp.get("lever_counters_moved"):
            # The lever ran and the outcome did not move. Still not promotable, and still not the
            # same as "the idea is worthless": tu93 shows the shape -- flag_resolved True, 288 nav
            # transitions observed, model_fitted False. The hypothesis class did not fit, so the
            # lever had nothing to act on. That is a statement about this corpus, not the lever.
            return False, (
                "UNINTERPRETABLE_FIRED_NO_EFFECT: the lever's own counters moved, so it IS wired "
                "and running, but every game returned byte-identical levels and actions. Read the "
                "row's fire_counters to tell 'fitted and found nothing to do' from 'never fitted' "
                "-- only the first is a reportable null about the lever's value."
            )
        return False, (
            "UNINTERPRETABLE_NO_EFFECT: every game returned byte-identical levels AND actions, "
            "and no lever counter moved either. The flag did not resolve on this engine. That is "
            "a WIRING result, not evidence about the capability. Do NOT record it as a null."
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


def sweep_candidates(engine: str, data: dict) -> tuple[list[str], dict[str, str]]:
    """(flags to measure, {flag: why-skipped}). Every exclusion is reported, never silent.

    A sweep that quietly narrows its own candidate list is the pattern-narrower-than-its-concept
    failure with a progress bar. Whatever it declines to touch, it says so and says why.
    """
    reach = reachable_flags(engine=engine)
    todo, skipped = [], {}
    for flag in discover_flags():
        entry = (data.get("flags") or {}).get(flag) or {}
        if entry.get("evidence"):
            skipped[flag] = "already measured"
        elif flag not in reach:
            skipped[flag] = f"not reachable on {engine}"
        else:
            kind = classify_flag(flag)
            if kind != "bool":
                skipped[flag] = f"{kind}: setting it to 1 would corrupt the run, not enable it"
            else:
                todo.append(flag)
    return todo, skipped


def cmd_sweep(engine: str, limit: int | None, dry_run: bool) -> int:
    """Measure every unmeasured boolean capability flag, promoting what earns it.

    THE SHARED BASELINE. `--measure` runs baseline+arm per flag. A sweep of 23 flags that way is 46
    sweeps. The search is DETERMINISTIC -- dc22 spent exactly 11,737 actions on two separate runs --
    so the baseline is identical every time and can be computed ONCE and reused. That is 24 sweeps
    instead of 46, and the halving is a consequence of the determinism, not a shortcut around it.

    If a stochastic component is ever added to the search, this optimisation becomes WRONG and the
    per-flag baseline has to come back. The same paragraph in this file's header says the same
    thing about the k-independent-runs rule; both assumptions live or die together.

    RESUMABLE by construction: a flag with recorded evidence is skipped, so an interrupted sweep
    continues where it stopped rather than re-measuring what it already knows.
    """
    data = load()
    todo, skipped = sweep_candidates(engine, data)
    print(f"arc-flag-ledger sweep on `{engine}`: {len(todo)} candidate(s), {len(skipped)} skipped")
    reasons: dict[str, int] = {}
    for why in skipped.values():
        reasons[why.split(":")[0]] = reasons.get(why.split(":")[0], 0) + 1
    for why, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"    skipped {n:>3}  {why}")
    if limit:
        todo = todo[:limit]
    if dry_run or not todo:
        for f in todo:
            print(f"    would measure {f}")
        return 0

    tmp = Path(os.environ.get("TMPDIR", "/tmp"))
    print("\n  shared baseline sweep (deterministic, computed once)...")
    base = run_bench(out=tmp / "arc_sweep_base.json", engine=engine)
    print(
        f"    baseline: {base['total_levels_cleared']} level(s), {base['total_actions_spent']} actions"
    )

    promoted = []
    for i, flag in enumerate(todo, 1):
        print(f"\n  [{i}/{len(todo)}] {flag}")
        arm = run_bench({flag: "1"}, out=tmp / "arc_sweep_arm.json", engine=engine)
        cmp = compare(base, arm)
        ok, why = verdict(cmp)
        print(f"    {arm['total_levels_cleared']} level(s), {arm['total_actions_spent']} actions")
        print(f"    {why}")
        data = load()  # re-read: a concurrent --discover may have touched the file
        entry = data.setdefault("flags", {}).setdefault(
            flag, {"state": "unevaluated", "evidence": []}
        )
        entry.setdefault("evidence", []).append(
            {"date": _now(), "value": "1", "verdict": why, "engine": engine, **cmp}
        )
        entry["last_measured"] = _now()
        entry["promotable"] = ok
        entry["benchmark_reachable"] = True
        entry["measured_on_engine"] = engine
        if ok:
            entry["state"] = "on"
            entry["promoted_on"] = _now()
            promoted.append(flag)
            print("    PROMOTED -> default ON")
        save(data)

    print(f"\n  swept {len(todo)}, promoted {len(promoted)}: {promoted or 'none'}")
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
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="measure every unmeasured boolean capability flag on the chosen engine, promoting "
        "what earns it. Resumable; skips anything already measured.",
    )
    ap.add_argument("--limit", type=int, default=None, help="sweep only the first N candidates")
    ap.add_argument("--dry-run", action="store_true", help="sweep: list candidates, measure none")
    args = ap.parse_args(argv)

    if args.discover:
        return cmd_discover()
    if args.measure:
        return cmd_measure(args.measure, args.value, force=args.force, engine=args.engine)
    if args.sweep:
        return cmd_sweep(args.engine, args.limit, args.dry_run)
    if args.promote:
        return cmd_promote(args.promote)
    return cmd_status()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
