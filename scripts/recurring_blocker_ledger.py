#!/usr/bin/env python3
"""A blocker that recurs is a task nobody is doing. Find them and escalate.

WHY THIS EXISTS. Unattended operation fails differently from attended operation: a failure
nobody reads never gets fixed, so the same one recurs indefinitely and the loop keeps paying for
it. Measured 2026-08-13 over the last 14 milestones: 54 blocked tasks, and **28 of them carry
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

WHAT THE BLOCKERS ACTUALLY SAY, and a correction to this file's own first claim. The first
version of this tool reported that 37 of 58 blocked artifacts recorded no reason at all. That was
wrong, and wrong in this project's signature way: `_REASON_FIELDS` omitted `gate_check_summary`,
the field the conductor's own pre-gate writes. Re-measured with the corrected list, 48 of 54
blocked artifacts DO record a reason and only 6 record nothing. The tool built to catch recurring
blockers had shipped with a pattern list narrower than its concept.

The corrected reading is also more useful. Of the 28 recurring blocks, 9 say the upstream
artifact was never found, 4 say the gated field read `None`, and 15 say the upstream readiness
score was 0 when 1 was expected. Only the last group is a gate doing its job. The first two are a
BROKEN CONTRACT between planner and agent: the planner writes `gated_on: <task>.<field>` and the
agent's artifact does not carry that field name. The near-misses are unmistakable -- gated on
`scorer_ready` where the artifact wrote `ebcn_scorer_ready`, on `pwa_ready` where it wrote
`pwa_kan_ready`, on `ledger_ready` where it wrote `cerce_ledger_ready`. Nothing reconciles the two
names, so the gate reads None and blocks a task that had no reason to be blocked.

`--reasons` still reports diagnostic coverage, because a blocker nobody can diagnose is one you
are guaranteed to repeat. It just is not this corpus's problem.

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

# Fields an artifact might use to say WHY it blocked.
#
# WIDENED 2026-08-13, and the reason matters more than the list. The first version omitted
# `gate_check_summary` -- the field the conductor's OWN pre-gate writes, and the one most blocked
# artifacts actually use. It reported "37 of 58 record nothing". The true number is 6 of 54. The
# tool built to find recurring blockers had the project's own recurring bug: a pattern list
# narrower than the concept it named. Corrected before the wrong number was acted on.
#
# Keep this list wider than seems necessary. Over-counting a reason is a small error; declaring a
# diagnosed corpus undiagnosed sends someone to re-run 31 tasks that already told them why.
_REASON_FIELDS = (
    "gate_check_summary",  # written by the conductor pre-gate; the dominant field
    "gate_check_results",
    "blocked_at_layer",
    "stall_details",
    "block_reason",
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
            groups[normalise(v)].append((str(m.get("id", "?")), p.name, _reason_of(art)))
            coverage["with_reason" if any(art.get(f) for f in _REASON_FIELDS) else "no_reason"] += 1
    return dict(groups), coverage, n_blocked


def _reason_of(art: dict) -> str:
    """The first recorded reason, normalised to its SHAPE.

    Grouping by verdict alone is close to useless here: 28 blocks share the identical verdict
    `blocked_gate_check_failed`, and that name says nothing about what to fix. The reason text is
    what separates "a dependency never landed" from "a gate refused genuinely unready work". Ids
    and numbers are stripped so the shapes group together.
    """
    for f in _REASON_FIELDS:
        v = art.get(f)
        if not v:
            continue
        s = v if isinstance(v, str) else json.dumps(v)[:200]
        # `exp` must be followed by a DIGIT. Without that this ate the word "expected" and
        # rendered every gate line as `actual=N == <task>=N`, hiding the comparison it exists to
        # show -- the same substring bug CLAUDE.md records for `diffusiongemma_met` in "meta".
        s = re.sub(r"\bexp\d[\w-]*", "<task>", s, flags=re.I)
        s = re.sub(r"\d+", "N", s)
        return re.sub(r"\s+", " ", s).strip()[:110]
    return ""


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
        ms = sorted({m for m, _, _ in hits})
        lines.append(
            f"- **x{len(hits)}** `{key}` — milestones {ms[0]}..{ms[-1]}; e.g. `{hits[0][1]}`"
        )
        for reason, n in Counter(r for _, _, r in hits if r).most_common(2):
            lines.append(f"  - x{n} because: `{reason}`")
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
        ms = sorted({m for m, _, _ in hits})
        print(f"    x{len(hits):<3} {key}")
        print(f"          milestones {ms[0]}..{ms[-1]}  e.g. {hits[0][1]}")
        # The verdict name alone does not say what to fix. Show what the artifacts SAID.
        for reason, n in Counter(r for _, _, r in hits if r).most_common(2):
            print(f"          x{n} because: {reason}")

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
