#!/usr/bin/env python3
"""Adversarial claim-refutation audit: what would REFUTE this claim, and was that checked?

WHY A SIXTH AUDIT (REQ-OPS-CLAIM-REFUTATION-6650). `adversarial_verify.py` is mechanical
and catches fabrication. The five milestone-close AI audits each cover a narrow surface:
verifier code, QA guards, the landing page, artifact recording conventions, ARC solve
provenance. None reviews an ordinary experiment's CLAIM. Four real misses from 2026-08
motivated this audit; each passed every mechanical check and was caught only by a human:

  1. exp6478 reported a positive selection result that was a TAUTOLOGY of the corpus
     construction -- success and energy were computed from the same violated-constraint
     list, so argmin-energy could not fail, and a plain violation-count comparator TIED it.
  2. A metric named `heldout_accuracy` measured IN-SAMPLE fit -- memorizing engines scored
     HIGHER -- yet conclusions cited it as generalization.
  3. A `.get()` on a key that does not exist on the rows read None as zero, and a
     "solves zero levels" claim propagated for a day.
  4. Rows marked invalid sat next to valid rows and were trivially averaged in.

THE PROMPT IS QUESTION-SHAPED, NOT A PATTERN LIST. A pattern list drifts narrower than its
concept (see the QA-Layer Authenticity Discipline in CLAUDE.md). The reviewer is asked ONE
question: name the observation that would refute the headline claim, then say whether the
artifact shows that refutation was given a real chance to happen.

WHAT IT DELEGATES. Degenerate-corpus / no-headroom detection already exists as
`adversarial_verify.check_false_negative_risk`, and circular gate-flips as
`check_circular_moat_overclaim`. This audit runs both as a mechanical pre-pass and hands
their flags to the reviewer as context. It does not duplicate them.

WHAT IT NEVER DOES. It never edits an artifact, never blocks a commit, and never fails the
conductor. Same contract as its five siblings: it surfaces, the operator decides.

THE AUDIT-INTEGRITY GUARD (Layer 1.5). A hostile reviewer can invent its smoking gun. Any
flagged verdict whose `## EVIDENCE` section quotes a token absent from the audited artifact
is downgraded to CANNOT_DETERMINE. Sections that discuss ABSENT things (`## WHAT WOULD
REFUTE IT`, `## WAS THAT CHECKED`) are exempt, per the qa-layer audit's 2026-07-29 lesson:
a refutation check the artifact lacks is, by definition, not in the artifact.

THE REPORT IS THE RECEIPT (REQ-CONDUCTOR-RECEIPT-1). The caller verifies the report file's
mtime advanced, never the exit code. The wall-clock budget (`--budget-seconds`) is checked
between artifacts; on exhaustion the audit stops and still writes a PARTIAL report naming
the completed count. A deadline the audit knows about produces a partial receipt; the
caller's kill timeout produces nothing.

Usage:
    python3 scripts/experiment_claim_audit.py --recent 8
    python3 scripts/experiment_claim_audit.py --artifact results/experiment_6478_*.json
    python3 scripts/experiment_claim_audit.py --recent 8 --dry-run   # no LLM call
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "ops" / "experiment_claim_audit_report.md"

VERDICTS = (
    "CLAIM_SUPPORTED",
    "CLAIM_OVERSTATED",
    "CLAIM_REFUTED_BY_OWN_DATA",
    "NO_CLAIM",
    "CANNOT_DETERMINE",
)
# Verdicts that accuse; only these trigger the integrity-guard evidence sweep.
FLAGGED_VERDICTS = ("CLAIM_OVERSTATED", "CLAIM_REFUTED_BY_OWN_DATA")
RECOMMENDATIONS = ("KEEP", "NARROW_CLAIM", "ADD_MISSING_CONTROL", "CORRECT_THE_RECORD")

PROMPT = """\
You are a HOSTILE CLAIMS REVIEWER. You did not run this experiment and you cannot re-run
it. You have only the artifact below plus mechanical pre-pass notes. Assume the numbers
are honest -- fabrication was already screened by a different tool. Your target is the
CLAIM, not the data.

Answer ONE question: what observation, if it appeared in this artifact's OWN data, would
REFUTE the headline claim -- and does the artifact show that refutation was given a real
chance to happen?

Work through these questions in order. They are questions, not patterns to grep for.

1. COULD IT HAVE FAILED? Trace how the success criterion and the method under test are
   computed. If both derive from the same quantity, or the corpus is built so the method
   cannot lose, the result is true by construction and supports nothing. A comparator arm
   that exactly TIES the method is strong evidence the method adds nothing over it.
   A positive verdict on a result that could not have come out otherwise is
   CLAIM_OVERSTATED no matter how carefully the sentence is worded. Downstream
   aggregation reads the verdict token (`complete_positive`), not the hedges around it.
2. WHO IS THE ORACLE? If the verifier IS the check that defines correctness
   (`verifier_is_oracle` true), a positive claim about the verifier's added value is
   circular. The measurement may stand as execution-grounded; the VALUE claim does not.
3. WHAT DID THE MODEL SEE? If a metric is named held-out or generalization, ask whether
   the scored data was shown to the model or engine that produced the candidate. An
   in-sample fit cited as generalization is unsupported, whatever the field is named.
4. WHAT WOULD A REAL RIVAL DO? Beating a "first", "random" or "shuffled" arm is a sanity
   check, not evidence of value. Name the cheapest serious baseline the claim would need
   to beat. If that baseline is present and ties or wins, say so plainly.
5. WHICH ROWS COUNT? If any rows carry a validity flag that is false, does the headline
   exclude them? Do the fields the headline cites actually exist on the rows they are
   read from? A missing key silently read as zero is a classic wrong-key error.

The mechanical pre-pass notes cover degenerate-corpus and no-headroom signals. Trust
them; do not re-derive them.

Verdict meanings. CLAIM_SUPPORTED: refutation was genuinely POSSIBLE, was checked, and
did not occur. CLAIM_OVERSTATED: the data is real but the design cannot support the
claim as it will be consumed -- true by construction, a circular value claim, in-sample
cited as generalization, or wins only over sanity-check arms while a trivial comparator
ties. CLAIM_REFUTED_BY_OWN_DATA: the artifact's own rows contradict the headline.

Do NOT invent a problem. An honest null backed by rows is CLAIM_SUPPORTED. A scaffolding
or receipt artifact making no comparative claim is NO_CLAIM. Disliking the design is not
a finding; a refutation that was possible, checked, and absent is CLAIM_SUPPORTED.

In `## EVIDENCE`, quote verbatim only field names or values that literally appear in the
artifact text, each in backticks. Put anything the artifact LACKS in `## WHAT WOULD
REFUTE IT` or `## WAS THAT CHECKED`, never in `## EVIDENCE`.

Reply in this exact format:

## VERDICT
<one of: CLAIM_SUPPORTED | CLAIM_OVERSTATED | CLAIM_REFUTED_BY_OWN_DATA | NO_CLAIM | \
CANNOT_DETERMINE>

## THE HEADLINE CLAIM
<one sentence, or "no claim">

## WHAT WOULD REFUTE IT
<the concrete observation that would falsify the claim. May name checks the artifact lacks.>

## WAS THAT CHECKED
<yes/no, and where. May name checks the artifact lacks.>

## EVIDENCE
<verbatim backticked quotes from the artifact supporting your verdict. If none, write "none">

## RECOMMENDATION
<one of: KEEP | NARROW_CLAIM | ADD_MISSING_CONTROL | CORRECT_THE_RECORD>
"""


def _now() -> float:
    """Monotonic clock behind the budget, wrapped so tests can drive it."""
    return time.monotonic()


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
            input=f"{prompt}\n\n{body}",
            capture_output=True,
            text=True,
            timeout=420,
            check=False,
        )
        return (r.returncode == 0, r.stdout or r.stderr)
    except Exception as exc:  # noqa: BLE001
        return False, repr(exc)[:200]


def condense(obj: object, list_head: int = 3, max_str: int = 400) -> object:
    """Shrink an artifact for review without losing its claim-bearing fields.

    Artifacts run to megabytes because of per-unit row lists. Head-truncating the
    raw JSON would cut off whatever fields sort late, so instead long lists keep
    their first few entries plus an elision marker, and long strings are clipped.
    Top-level claim fields (verdict, arms, gates) always survive.
    """
    if isinstance(obj, dict):
        return {k: condense(v, list_head, max_str) for k, v in obj.items()}
    if isinstance(obj, list):
        if len(obj) > list_head * 2 + 2:
            head = [condense(x, list_head, max_str) for x in obj[:list_head]]
            return [*head, f"... {len(obj) - list_head} more entries elided ..."]
        return [condense(x, list_head, max_str) for x in obj]
    if isinstance(obj, str) and len(obj) > max_str:
        return obj[:max_str] + f"...({len(obj)} chars)"
    return obj


def prepass_notes(d: dict) -> list[str]:
    """Mechanical pre-pass: delegate to adversarial_verify's existing detectors.

    Degenerate-corpus / no-headroom and circular gate-flip detection already
    live in adversarial_verify.py. Run them here and hand the flags to the
    reviewer as context instead of re-implementing the logic.
    """
    notes: list[str] = []
    for key in ("honest_verdict", "verifier_is_oracle", "inference_substrate"):
        if key in d:
            notes.append(f"{key} = {json.dumps(d[key])[:200]}")
    try:
        if str(REPO / "scripts") not in sys.path:
            sys.path.insert(0, str(REPO / "scripts"))
        import adversarial_verify as av  # noqa: PLC0415

        flags: list = []
        av.check_false_negative_risk(d, flags)
        av.check_circular_moat_overclaim(d, flags)
        notes += [f"{f.kind}({f.severity}): {f.detail}"[:400] for f in flags]
        if not flags:
            notes.append("adversarial_verify pre-pass: no FALSE_NEGATIVE_RISK / circular flags")
    except Exception as exc:  # noqa: BLE001
        notes.append(f"adversarial_verify pre-pass unavailable: {exc!r}"[:200])
    return notes


def parse_verdict(report: str) -> str:
    m = re.search(r"##\s*VERDICT\s*\n+\s*([A-Z_]+)", report)
    v = (m.group(1) if m else "").strip()
    return v if v in VERDICTS else "UNKNOWN"


def parse_recommendation(report: str) -> str:
    m = re.search(r"##\s*RECOMMENDATION\s*\n+\s*([A-Z_]+)", report)
    v = (m.group(1) if m else "").strip()
    return v if v in RECOMMENDATIONS else "NONE"


def verify_quoted_evidence(report: str, body: str) -> list[str]:
    """Return backticked EVIDENCE tokens the reviewed body does NOT contain.

    The audit-integrity guard (Layer 1.5). Only `## EVIDENCE` is swept: the
    refutation sections are asked to describe what the artifact LACKS, and an
    absent check is correctly absent from the body (the qa-layer audit's
    2026-07-29 SILENT_NON_FIRING exemption, applied here from day one).
    """
    sec = re.search(r"##\s*EVIDENCE\s*\n(.*?)(?=\n##|\Z)", report, re.S)
    if not sec:
        return []
    bad = []
    for span in re.findall(r"`([^`\n]{5,120})`", sec.group(1)):
        if span in body:
            continue
        # A quoted span may join a field name and its value with punctuation the
        # JSON dump renders differently. Fall back to its word-tokens: the span
        # counts as present when every token of length >= 5 appears in the body.
        toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]{4,}", span)
        if toks and all(t in body for t in toks):
            continue
        bad.append(span)
    return bad


def _artifacts(n: int) -> list[Path]:
    res = REPO / "results"
    return sorted(res.glob("experiment_*.json"), key=lambda p: p.stat().st_mtime)[-n:]


def _write_report(
    rows: list[tuple[str, str, str, list[str]]],
    counts: dict[str, int],
    partial_note: str,
) -> None:
    lines = [
        "# Experiment claim-refutation audit",
        "",
        "One question per artifact: what would REFUTE the headline claim, and was that",
        "checked? Fabrication is out of scope (adversarial_verify covers it); this audit",
        "targets claims that are true by construction, circular, in-sample, baseline-weak,",
        "or contradicted by their own rows.",
        "",
        "This audit never edits an artifact and never blocks anything. It surfaces; the",
        "operator decides. Verdicts downgraded to CANNOT_DETERMINE by the audit-integrity",
        "guard rest on evidence the reviewer could not have read -- do NOT act on them.",
        "",
    ]
    if partial_note:
        lines += [partial_note, ""]
    lines += ["| verdict | count |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in counts.items() if v]
    lines.append("")
    for name, verdict, text, invented in rows:
        lines += [f"## {name}", "", f"**{verdict}**", ""]
        if invented:
            lines += [
                f"> Audit-integrity guard: quoted evidence {invented} does not appear in "
                "the artifact, so this verdict was downgraded and must not be acted on.",
                "",
            ]
        if text:
            lines += [text.strip(), ""]
    REPORT.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent", type=int, default=8)
    ap.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="explicit artifact path(s); overrides --recent",
    )
    ap.add_argument("--agent-type", default="codex")
    ap.add_argument("--model-name", default="gpt-5.5")
    ap.add_argument(
        "--budget-seconds",
        type=float,
        default=750.0,
        help="Wall-clock budget. On exhaustion the audit stops reviewing further "
        "artifacts and still writes a PARTIAL report (the receipt). 0 disables.",
    )
    ap.add_argument("--dry-run", action="store_true", help="list targets; make no LLM call")
    args = ap.parse_args(argv)

    targets = [Path(a) for a in args.artifact] if args.artifact else _artifacts(args.recent)
    if args.dry_run:
        print(f"experiment-claim-audit: {len(targets)} target(s)")
        for p in targets:
            print(f"  {p.name}")
        return 0

    deadline = _now() + args.budget_seconds if args.budget_seconds > 0 else None
    rows: list[tuple[str, str, str, list[str]]] = []
    counts = dict.fromkeys((*VERDICTS, "UNKNOWN", "SKIPPED_ALREADY_FLAGGED"), 0)
    completed = 0
    for i, p in enumerate(targets):
        if deadline is not None and _now() >= deadline:
            # Budget exhausted between artifacts: record the rest un-reviewed and
            # stop. The PARTIAL report below is still written -- the receipt must
            # exist even when the budget dies early (REQ-CONDUCTOR-RECEIPT-1).
            print(f"  [budget] exhausted after {completed}/{len(targets)} artifact(s)")
            rows += [(q.name, "NOT_REVIEWED_BUDGET", "", []) for q in targets[i:]]
            break
        try:
            data = json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            rows.append((p.name, "CANNOT_DETERMINE", f"unreadable: {exc!r}"[:160], []))
            counts["CANNOT_DETERMINE"] += 1
            completed += 1
            continue
        if isinstance(data, dict) and data.get("flagged_adversarial") is True:
            # Already quarantined by the fabrication gate; headline aggregation
            # skips it per CLAUDE.md, so a reviewer call here buys nothing.
            rows.append((p.name, "SKIPPED_ALREADY_FLAGGED", "", []))
            counts["SKIPPED_ALREADY_FLAGGED"] += 1
            completed += 1
            continue
        notes = prepass_notes(data) if isinstance(data, dict) else []
        body = json.dumps(condense(data), indent=1)[:60000]
        packet = (
            "---MECHANICAL PRE-PASS NOTES---\n"
            + "\n".join(notes)
            + f"\n\n---ARTIFACT {p.name}---\n"
            + body
        )
        ok, review = _call(args.agent_type, args.model_name, PROMPT, packet)
        if not ok:
            rows.append((p.name, "CANNOT_DETERMINE", f"reviewer call failed: {review[:160]}", []))
            counts["CANNOT_DETERMINE"] += 1
            completed += 1
            continue
        verdict = parse_verdict(review)
        invented = verify_quoted_evidence(review, body) if verdict in FLAGGED_VERDICTS else []
        if invented:
            verdict = "CANNOT_DETERMINE"
        counts[verdict] = counts.get(verdict, 0) + 1
        rows.append((p.name, verdict, review, invented))
        completed += 1
        print(f"  [{verdict:26}] {p.name}  ({parse_recommendation(review)})")

    partial_note = ""
    if completed < len(targets):
        partial_note = (
            f"**PARTIAL RUN** -- wall-clock budget {args.budget_seconds:.0f}s exhausted "
            f"after {completed} of {len(targets)} artifact(s)."
        )
    _write_report(rows, counts, partial_note)
    shown = REPORT.relative_to(REPO) if REPORT.is_relative_to(REPO) else REPORT
    print(f"\nexperiment-claim-audit: wrote {shown}")
    print(f"  { ({k: v for k, v in counts.items() if v}) }")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
