#!/usr/bin/env python3
"""Did a milestone MOVE anything, or did it produce artifacts?

WHY THIS EXISTS. `ops/north-star.md` §1 already states the test: "a milestone that produces a
new version of an existing artifact without moving the headline is noise". Nothing computes it.
Measured 2026-08-13 across milestones .544-.550: 193 commits, 93 artifacts, roughly 14
substantive measurements, and ZERO movement in `reproducible_total_levels` (183), games (25) or
the Kaggle score (0.09). That ratio is not reported anywhere, so nothing pushes back on it.

WHAT IT REPORTS, per milestone: how many artifacts carried a verdict, and how those verdicts
split between SUBSTANTIVE measurements and the three kinds of not-a-measurement this corpus
actually produces -- readiness/shadow/audit scaffolding, blocked/pre-gated tasks, and declared
nulls. Then the headline metrics, so "did anything move" is answerable in one line.

WHAT IT IS NOT. It is not a gate and never blocks. Scaffolding, blocked tasks and honest nulls
are all legitimate work -- a milestone of nulls that closes a real question is worth more than a
milestone of positives that re-measures a solved one. This tool does not judge; it makes the mix
visible so a human can. Turning it into a gate would reward relabelling verdicts, which is
exactly the failure mode `verdict_row_consistency_lint.py` exists to catch.

COVERAGE IS ALWAYS PRINTED. An artifact whose verdict cannot be classified is counted as
UNCLASSIFIED, never silently dropped into a category. A summary that hides how much it could not
read is the guard-is-green-while-blind state this project keeps finding in its own tools.

Usage:
    python3 scripts/milestone_progress_ledger.py                # last 8 milestones
    python3 scripts/milestone_progress_ledger.py --last 20
    python3 scripts/milestone_progress_ledger.py --milestone 2026.08.550
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMPLETE = REPO / "research-complete.yaml"
REGISTRY = REPO / "ops" / "arc_solve_registry.yaml"

# Verdict shapes this corpus actually emits, measured rather than guessed. Order matters: a
# verdict is classified by the FIRST category it matches, so blocked/null are tested before
# scaffolding, since a blocked readiness task is blocked first.
_BLOCKED = ("blocked", "skipped", "pre_gate", "gate_block", "not_run", "unusable")
_NULL = ("null", "no_reliable", "no_improvement", "no_delta", "no_value", "not_met", "negative")
_SCAFFOLD = (
    "ready",
    "shadow",
    "audit",
    "no_solve_claim",
    "default_off",
    "freeze",
    "preflight",
    "reconcil",
    "handoff",
    "no_scope_change",
    "manifest",
    "archive",
    "activate",
    "capstone",
    # Added 2026-08-13 after a spot-check: milestone-TRANSITION tasks were landing in
    # "substantive" and inflating the share. Their verdicts read like results ("V540 exact
    # states and V541 roadmap contracts validated") but they move a milestone forward rather
    # than measure anything. Erring toward scaffolding is deliberate -- this ledger exists to
    # resist a flattering count, so an ambiguous verdict should not be scored as a measurement.
    # "licens" covers the held-factor licensing family (freeze / license-matrix / qualified-cell
    # tasks). Verified narrow: only 2 verdicts in the recent corpus contain it and both were
    # already scaffolding by another marker, so it adds no new reclassification risk.
    "licens",
    "transition",
    "roadmap contract",
    "exact states",
    "terminal evidence",
    "validated;",
)


def _verdict(d: dict) -> str:
    v = d.get("honest_verdict")
    if isinstance(v, dict):
        v = v.get("value")
    return str(v or "")


def classify(verdict: str) -> str:
    v = verdict.lower()
    if not v:
        return "no_verdict"
    if any(t in v for t in _BLOCKED):
        return "blocked"
    if any(t in v for t in _NULL):
        return "null"
    if any(t in v for t in _SCAFFOLD):
        return "scaffolding"
    if v.startswith(("complete", "success", "passed", "shipped")):
        return "substantive"
    return "unclassified"


def _load_milestones() -> list[dict]:
    """Parse research-complete.yaml without a YAML dependency at import time.

    Uses PyYAML when present; the file is large and hand-edited, so a parse failure is reported
    rather than silently yielding an empty ledger that would read as "no milestones".
    """
    try:
        import yaml
    except ImportError:
        print("milestone-ledger: PyYAML not available; cannot read research-complete.yaml")
        return []
    try:
        d = yaml.safe_load(COMPLETE.read_text())
    except Exception as exc:  # noqa: BLE001
        print(f"milestone-ledger: research-complete.yaml unreadable ({exc}); nothing to report")
        return []
    ms = (d or {}).get("milestones") or []
    return [m for m in ms if isinstance(m, dict)]


def _artifact_for(task: dict) -> Path | None:
    dl = task.get("deliverable")
    if not isinstance(dl, str) or not dl.endswith(".json"):
        return None
    p = REPO / dl
    return p if p.exists() else None


def milestone_row(m: dict) -> dict:
    counts = {
        k: 0
        for k in ("substantive", "scaffolding", "blocked", "null", "unclassified", "no_verdict")
    }
    missing = 0
    for task in m.get("tasks") or []:
        if not isinstance(task, dict):
            continue
        p = _artifact_for(task)
        if p is None:
            missing += 1
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            counts["unclassified"] += 1
            continue
        counts[classify(_verdict(d))] += 1
    return {
        "id": m.get("id", "?"),
        "completed": m.get("completed", "?"),
        "n_tasks": len(m.get("tasks") or []),
        "artifact_missing": missing,
        **counts,
    }


def headline_state() -> dict:
    """The metrics north-star.md says a milestone must move to not be churn."""
    out: dict = {}
    try:
        for line in REGISTRY.read_text().splitlines():
            mm = re.match(r"^(reproducible_total_(?:levels|games)):\s*(\d+)", line)
            if mm:
                out[mm.group(1)] = int(mm.group(2))
    except Exception:  # noqa: BLE001
        pass
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--last", type=int, default=8)
    ap.add_argument("--milestone", default=None)
    args = ap.parse_args(argv)

    ms = _load_milestones()
    if not ms:
        return 0
    if args.milestone:
        ms = [m for m in ms if str(m.get("id")) == args.milestone]
    else:
        ms = ms[-args.last :]

    rows = [milestone_row(m) for m in ms]
    print(
        f"{'milestone':16} {'done':11} {'subst':>6} {'scaff':>6} {'block':>6} {'null':>5} {'?':>4} {'gone':>5}"
    )
    print("-" * 68)
    tot = {
        k: 0
        for k in (
            "substantive",
            "scaffolding",
            "blocked",
            "null",
            "unclassified",
            "artifact_missing",
        )
    }
    for r in rows:
        print(
            f"{r['id']:16} {str(r['completed']):11} {r['substantive']:>6} {r['scaffolding']:>6} "
            f"{r['blocked']:>6} {r['null']:>5} {r['unclassified'] + r['no_verdict']:>4} "
            f"{r['artifact_missing']:>5}"
        )
        for k in tot:
            tot[k] += r[k]

    total = sum(tot.values())
    print("-" * 68)
    print(
        f"{'TOTAL':16} {'':11} {tot['substantive']:>6} {tot['scaffolding']:>6} "
        f"{tot['blocked']:>6} {tot['null']:>5} {tot['unclassified']:>4} {tot['artifact_missing']:>5}"
    )
    if total:
        pct = 100.0 * tot["substantive"] / total
        print(f"\n  substantive share: {tot['substantive']}/{total} ({pct:.0f}%)")
    print(f"  headline metrics now: {headline_state() or 'unreadable'}")
    print(
        "\n  north-star.md §1: a milestone that produces a new version of an existing artifact\n"
        "  without moving the headline is noise. This ledger does not judge -- scaffolding,\n"
        "  blocked tasks and honest nulls are all legitimate work. It makes the mix visible so\n"
        "  a human can ask whether the headline moved, and if not, whether that was the plan."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
