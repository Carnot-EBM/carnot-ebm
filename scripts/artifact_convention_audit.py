#!/usr/bin/env python3
"""Adversarial audit: can a stranger CHECK this artifact's claim, or must they trust it?

WHY A FIFTH AUDIT, AND WHY ADVERSARIAL RATHER THAN A LINT. Four adversarial audits already run
at milestone close -- landing page, verifier authenticity, QA-layer guards, ARC solve provenance.
Every one of them audits CODE or DOCS. None audits the ARTIFACTS, which are the research record
and the thing every downstream claim is built on.

The gap was found the hard way. `scripts/verdict_row_consistency_lint.py` was written to catch
verdicts contradicted by their own rows. It took FIVE rounds -- it scanned only top-level keys
and missed its own founding case, discarded a metric because an exclusion matched `_n` mid-name,
printed "OK" while skipping 57 of 60 artifacts, could not see booleans, then manufactured noise
when booleans were added. The pattern-narrower-than-its-concept failure recurred five times
inside the tool built to catch that failure. That is the signature of a problem a pattern list
cannot solve: judging whether a claim is CHECKABLE is semantic, not syntactic.

An adversarial reviewer does not need to be taught every artifact shape. It needs one question:
**could someone who was not there verify this claim from what is written down?**

WHAT IT CHECKS. Two conventions, both measured, both already in the planner's REQUIRED ARTIFACT
FIELDS guidance (prevention) -- this is the detection half:

  1. A COMPARATIVE claim must record PER-UNIT ROWS. Measured 2026-08-13: 39 of the 60 most
     recent artifacts carry rows, 21 do not. A headline computed from a pooled mean alone cannot
     be rechecked without re-running the task.
  2. A BLOCKED verdict must record WHY, in `gate_check_summary`. Measured: 48 of 54 blocked
     artifacts already do, and only 6 record nothing.

BOTH NUMBERS ARE CORRECTIONS, and the corrections are the point. The first pass reported 3 of 60
for rows and 37 of 58 for reasons. Both were measurement bugs of exactly the kind this audit
hunts: the row check read only top-level keys while most rows sit nested one or two levels down,
and the reason check omitted `gate_check_summary`, the field the conductor's own pre-gate writes.
Two detectors, both blind, both reporting a corpus far worse than it is. Nothing was re-run on
the strength of those numbers, which is the only reason this is a correction and not an incident.

Keep the corrected numbers here. A reviewer primed with "57 of 60 are unfalsifiable" will hunt
for a crisis that is not there, and will find one.

WHAT IT NEVER DOES. It never edits an artifact, never blocks a commit, and never fails the
conductor. Same contract as its four siblings: it surfaces, the operator decides. A convention
audit with teeth would reward writing fields that satisfy a checker rather than fields that let
a reader check the work -- which is the failure this whole line of work is trying to avoid.

THE AUDIT-INTEGRITY GUARD (Layer 1.5) applies here as it does to the siblings. A hostile
reviewer can invent its smoking gun, so any flagged verdict whose quoted high-specificity
evidence does not literally appear in the audited artifact is downgraded to CANNOT_DETERMINE
rather than acted on.

Usage:
    python3 scripts/artifact_convention_audit.py --recent 12
    python3 scripts/artifact_convention_audit.py --recent 12 --agent-type codex
    python3 scripts/artifact_convention_audit.py --recent 12 --dry-run   # no LLM call
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "ops" / "artifact_convention_audit_report.md"

VERDICTS = ("CHECKABLE", "AGGREGATE_ONLY", "BLOCKED_WITHOUT_DIAGNOSTIC", "CANNOT_DETERMINE")

PROMPT = """\
You are a HOSTILE RESEARCH AUDITOR. You did not run this experiment and you cannot re-run it.
You have only the artifact below.

Answer ONE question: could you CHECK this artifact's headline claim from what is written down,
or would you have to TRUST it?

Two specific failures to hunt for.

1. AGGREGATE_ONLY. The artifact makes a COMPARATIVE claim (arm A beat arm B, a gate was met, a
   pooled mean improved) but records only aggregate numbers -- no per-unit rows giving the metric
   for each game, seed, cell or condition. Without rows, nobody can tell whether a pooled mean
   came from a broad effect or one outlier, whether a control was degenerate, or whether half
   the units had no headroom to move. Real examples from this corpus: a gate reported MET while
   the control arm was byte-identical to the baseline on 20 of 25 rows; another reported MET on
   one win, one loss and two units pinned at a floor and ceiling.

2. BLOCKED_WITHOUT_DIAGNOSTIC. The verdict says the task was blocked or gated, but the artifact
   does not record WHICH check failed and what value it saw. Such a blocker cannot be
   investigated without re-running the task, so it recurs.
   Most blocked artifacts in this corpus DO record a reason, usually in `gate_check_summary`. If
   you find one, the artifact passes this check -- do not flag it because the reason is terse.
   Flag only an artifact that says it was blocked and says nothing at all about what blocked it.

An artifact that records NO comparative claim and is NOT blocked is CHECKABLE by default -- do
not invent a problem. An honest null with per-unit rows is CHECKABLE. A scaffolding or receipt
artifact making no claim is CHECKABLE.

Reply in this exact format:

## VERDICT
<one of: CHECKABLE | AGGREGATE_ONLY | BLOCKED_WITHOUT_DIAGNOSTIC | CANNOT_DETERMINE>

## WHAT THE CLAIM IS
<one sentence: the artifact's headline claim, or "no claim">

## WHAT IS MISSING
<the exact field(s) a reader would need and cannot find. Quote field names that ARE present to
show what you looked at. If nothing is missing, write "nothing">

## THE CHECK A READER CANNOT DO
<one concrete question a reader cannot answer from this artifact. If none, write "none">
"""


def _call(agent: str, model: str, prompt: str, body: str) -> tuple[bool, str]:
    """Invoke the configured reviewer CLI. Mirrors the sibling audits' shape."""
    cmds = {
        "codex": ["codex", "exec", "--model", model, "-"],
        "claude": ["claude", "-p", "--model", model],
        "gemini": ["gemini", "-m", model, "-p", "-"],
    }
    cmd = cmds.get(agent)
    if not cmd:
        return False, f"unknown agent type {agent!r}"
    try:
        r = subprocess.run(
            cmd,
            input=f"{prompt}\n\n---ARTIFACT---\n{body}",
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        return (r.returncode == 0, r.stdout or r.stderr)
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)[:200]


def parse_verdict(report: str) -> str:
    m = re.search(r"##\s*VERDICT\s*\n+\s*([A-Z_]+)", report)
    v = (m.group(1) if m else "").strip()
    return v if v in VERDICTS else "UNKNOWN"


def verify_quoted_evidence(report: str, body: str) -> list[str]:
    """Return quoted field names the artifact does NOT contain.

    The audit-integrity guard (Layer 1.5). A hostile reviewer can invent the field it claims to
    have looked for, so a flagged verdict resting on invented evidence must be downgraded rather
    than acted on. Only the WHAT IS MISSING section is swept: the other sections are asked to
    describe what is ABSENT, and absent things are correctly not in the body.
    """
    sec = re.search(r"##\s*WHAT IS MISSING\s*\n(.*?)(?=\n##|\Z)", report, re.S)
    if not sec:
        return []
    bad = []
    for tok in re.findall(r"`([a-z_][a-z0-9_]{4,})`", sec.group(1)):
        # Only PRESENT-tense claims are checkable: the reviewer was asked to quote fields that
        # ARE present to show what it read. A field named as missing is expected to be absent.
        if tok in body:
            continue
        bad.append(tok)
    return bad


def _artifacts(n: int) -> list[Path]:
    res = REPO / "results"
    return sorted(res.glob("experiment_*.json"), key=lambda p: p.stat().st_mtime)[-n:]


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent", type=int, default=12)
    ap.add_argument("--agent-type", default="codex")
    ap.add_argument("--model-name", default="gpt-5.5")
    ap.add_argument("--dry-run", action="store_true", help="list targets; make no LLM call")
    args = ap.parse_args(argv)

    targets = _artifacts(args.recent)
    if args.dry_run:
        print(f"artifact-convention-audit: {len(targets)} target(s)")
        for p in targets:
            print(f"  {p.name}")
        return 0

    rows, counts = [], dict.fromkeys((*VERDICTS, "UNKNOWN"), 0)
    for p in targets:
        try:
            body = p.read_text()[:60000]
        except Exception as exc:  # noqa: BLE001
            rows.append((p.name, "CANNOT_DETERMINE", f"unreadable: {exc!r}"[:120], []))
            counts["CANNOT_DETERMINE"] += 1
            continue
        ok, report = _call(args.agent_type, args.model_name, PROMPT, body)
        if not ok:
            rows.append((p.name, "CANNOT_DETERMINE", "reviewer call failed", []))
            counts["CANNOT_DETERMINE"] += 1
            continue
        verdict = parse_verdict(report)
        invented = verify_quoted_evidence(report, body) if verdict in VERDICTS[1:3] else []
        if invented:
            verdict = "CANNOT_DETERMINE"
        counts[verdict] = counts.get(verdict, 0) + 1
        rows.append((p.name, verdict, report, invented))
        print(f"  [{verdict:26}] {p.name}")

    lines = [
        "# Artifact convention audit",
        "",
        "Can a stranger CHECK each artifact's claim, or must they trust it? Two conventions:",
        "a comparative claim records PER-UNIT ROWS; a blocked verdict records WHY.",
        "",
        "This audit never edits an artifact and never blocks anything. It surfaces; the operator",
        "decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on",
        "evidence the reviewer could not have read -- do NOT act on them.",
        "",
        "| verdict | count |",
        "|---|---|",
    ]
    lines += [f"| {k} | {v} |" for k, v in counts.items() if v]
    lines.append("")
    for name, verdict, report, invented in rows:
        lines += [f"## {name}", "", f"**{verdict}**", ""]
        if invented:
            lines += [
                f"> Audit-integrity guard: quoted field(s) {invented} do not appear in the "
                "artifact, so this verdict was downgraded and must not be acted on.",
                "",
            ]
        lines += [report.strip(), ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nartifact-convention-audit: wrote {REPORT.relative_to(REPO)}")
    print(f"  {counts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
