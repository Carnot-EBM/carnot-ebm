#!/usr/bin/env python3
"""A blocker that recurs is a task nobody is doing. Find them and escalate.

WHY THIS EXISTS. Unattended operation fails differently from attended operation: a failure
nobody reads never gets fixed, so the same one recurs indefinitely and the loop keeps paying for
it. Measured 2026-08-13 over the last 14 milestones: 58 blocked tasks, and **31 of them carry
the identical verdict `blocked_gate_check_failed`** -- one blocker, 31 times, across months,
with nothing escalating it. Blocked is also the single largest category of all work (31% of 108
tasks, `scripts/milestone_progress_ledger.py`).

The same shape showed up in the publish path the same day: a stale marker in
`prep_daily_submission.py` aborted EVERY dataset publish for six days while blaming the wrong
thing. Attended, someone investigates on day one. Unattended, it just accumulates.

WHAT THIS DOES. Groups blocked verdicts across completed milestones by normalised message,
reports any that recur at or above a threshold, and can APPEND a MANDATORY-NEXT-MILESTONE entry
to `ops/known-issues.md`. That last part is deliberate reuse: the Overdue-Priority Forcing
Function already makes the planner pick up entries pending 3+ milestones, so escalation plugs
into machinery that exists rather than adding a new enforcement path.

IT DOES NOT BLOCK ANYTHING. A recurring blocker may be correct -- a gate that keeps refusing
genuinely unready work is doing its job. What is not acceptable is nobody LOOKING. This makes
the recurrence visible and lets the existing priority mechanism schedule the investigation.

A SECOND FINDING IT SURFACES. Of the 31 `blocked_gate_check_failed` artifacts, ZERO recorded
why the gate failed -- no `gate_reason`, no `failed_gates`, nothing. A blocked verdict with no
diagnostic cannot be investigated later without re-running the task. `--reasons` reports that
coverage, because a blocker you cannot diagnose is one you are guaranteed to repeat.

Usage:
    python3 scripts/recurring_blocker_ledger.py
    python3 scripts/recurring_blocker_ledger.py --min 3 --window 20
    python3 scripts/recurring_blocker_ledger.py --emit-known-issue
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
COMPLETE = REPO / "research-complete.yaml"
KNOWN_ISSUES = REPO / "ops" / "known-issues.md"

# Fields an artifact might use to say WHY it blocked. Deliberately broad: the point of the
# coverage number is to be honest about how many artifacts record nothing at all.
_REASON_FIELDS = (
    "gate_reason",
    "gate_check",
    "blocked_reason",
    "failed_gates",
    "gate_failures",
    "reason",
    "blocker",
    "precondition_failures",
    "preconditions_checked",
)


def _verdict(d: dict) -> str:
    v = d.get("honest_verdict")
    if isinstance(v, dict):
        v = v.get("value")
    return str(v or "")


def normalise(verdict: str) -> str:
    """Collapse a verdict to its blocker IDENTITY.

    Numbers, experiment ids and milestone versions are stripped, so
    `blocked: Exp6262 readiness controls did not pass` and
    `blocked: Exp6301 readiness controls did not pass` count as ONE recurring blocker rather
    than two singletons. Without this, per-task ids hide every recurrence.
    """
    v = verdict.lower().strip()
    v = re.sub(r"\bexp\s*\d+\b", "exp", v)
    v = re.sub(r"\bv\d{3,}\b", "v", v)
    v = re.sub(r"\d{4}[.-]\d{2}[.-]\d+", "", v)
    v = re.sub(r"\d+(\.\d+)?", "", v)
    return re.sub(r"\s+", " ", v).strip()[:90]


def _is_blocked(verdict: str) -> bool:
    v = verdict.lower()
    return any(t in v for t in ("blocked", "gate_block", "pre_gate", "skipped"))


def collect(window: int) -> tuple[dict[str, list], Counter, int]:
    """Return (blocker -> [(milestone, artifact)], reason-coverage counter, n_blocked)."""
    try:
        import yaml
    except ImportError:
        print("recurring-blocker-ledger: PyYAML unavailable; cannot read research-complete.yaml")
        return {}, Counter(), 0
    try:
        d = yaml.safe_load(COMPLETE.read_text()) or {}
    except Exception as exc:  # noqa: BLE001
        print(f"recurring-blocker-ledger: research-complete.yaml unreadable ({exc})")
        return {}, Counter(), 0

    groups: dict[str, list] = defaultdict(list)
    coverage: Counter = Counter()
    n_blocked = 0
    for m in (d.get("milestones") or [])[-window:]:
        if not isinstance(m, dict):
            continue
        for t in m.get("tasks") or []:
            if not isinstance(t, dict):
                continue
            dl = t.get("deliverable")
            if not isinstance(dl, str) or not dl.endswith(".json"):
                continue
            p = REPO / dl
            if not p.exists():
                continue
            try:
                art = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            v = _verdict(art)
            if not _is_blocked(v):
                continue
            n_blocked += 1
            groups[normalise(v)].append((str(m.get("id", "?")), p.name))
            coverage["with_reason" if any(art.get(f) for f in _REASON_FIELDS) else "no_reason"] += 1
    return dict(groups), coverage, n_blocked


def emit_known_issue(recurring: list[tuple[str, list]]) -> bool:
    """Append ONE dated MANDATORY entry naming every over-threshold blocker.

    Appends rather than rewrites (never-prune), and writes a single entry rather than one per
    blocker so the priorities list does not get flooded by a mechanical process.
    """
    if not recurring:
        return False
    today = datetime.now(UTC).strftime("%Y-%m-%d")
    lines = [
        "",
        f"## {today} MANDATORY-NEXT-MILESTONE: recurring blockers nobody has investigated",
        "",
        "Emitted by `scripts/recurring_blocker_ledger.py`. Each line is ONE blocker message that",
        "has stopped work repeatedly across milestones. A recurring blocker may be correct -- a",
        "gate refusing genuinely unready work is doing its job -- but nothing has LOOKED, which is",
        "the unattended failure mode: a failure nobody reads never gets fixed.",
        "",
    ]
    for key, hits in recurring:
        ms = sorted({m for m, _ in hits})
        lines.append(
            f"- **x{len(hits)}** `{key}` — milestones {ms[0]}..{ms[-1]}; e.g. `{hits[0][1]}`"
        )
    lines += [
        "",
        "Investigate the highest-count blocker first: diagnose the root cause, then either fix it",
        "or record why the block is correct and expected. Do NOT simply re-run the task.",
        "",
    ]
    with KNOWN_ISSUES.open("a", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return True


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min", type=int, default=3, help="recurrence threshold")
    ap.add_argument("--window", type=int, default=14, help="how many recent milestones to scan")
    ap.add_argument("--emit-known-issue", action="store_true")
    ap.add_argument("--reasons", action="store_true", help="report diagnostic coverage only")
    args = ap.parse_args(argv)

    groups, coverage, n_blocked = collect(args.window)
    if not n_blocked:
        print("recurring-blocker-ledger: no blocked tasks in window")
        return 0

    no_reason = coverage.get("no_reason", 0)
    print(
        f"recurring-blocker-ledger: {n_blocked} blocked task(s) across {args.window} milestones, "
        f"{len(groups)} distinct blocker(s)."
    )
    print(
        f"  diagnostic coverage: {coverage.get('with_reason', 0)} record a reason, "
        f"{no_reason} record NOTHING."
    )
    if no_reason:
        print(
            "  A blocked verdict with no diagnostic cannot be investigated without re-running\n"
            "  the task, which is how the same blocker recurs indefinitely."
        )
    if args.reasons:
        return 0

    recurring = sorted(
        ((k, v) for k, v in groups.items() if len(v) >= args.min),
        key=lambda kv: -len(kv[1]),
    )
    if not recurring:
        print(f"  no blocker recurs {args.min}+ times.")
        return 0

    print(f"\n  RECURRING ({args.min}+ times):")
    for key, hits in recurring:
        ms = sorted({m for m, _ in hits})
        print(f"    x{len(hits):<3} {key}")
        print(f"          milestones {ms[0]}..{ms[-1]}  e.g. {hits[0][1]}")

    if args.emit_known_issue:
        if emit_known_issue(recurring):
            print(
                f"\n  Appended a MANDATORY-NEXT-MILESTONE entry to {KNOWN_ISSUES.name}. The "
                "Overdue-Priority\n  Forcing Function will surface it to the planner."
            )
    else:
        print("\n  Re-run with --emit-known-issue to schedule these for investigation.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
