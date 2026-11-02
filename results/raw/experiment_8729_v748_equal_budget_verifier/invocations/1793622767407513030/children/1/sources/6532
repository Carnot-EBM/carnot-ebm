#!/usr/bin/env python3
"""Cheap contradiction detectors that ESCALATE to an adversarial reviewer when they fire.

WHY THIS EXISTS (2026-08-16 operator directive, after a session where manual second opinions
repeatedly found what the loop's own machinery missed).

Three independent reviews during one experiment found: a prompt directive specifying the wrong
engine arity, which silently zeroed every cell that obeyed it; a "held-out" accuracy that was
scored against the very transitions the model was shown; and a repetition-control lever that was
gated off for the whole run. Each cost hours. Each was found by a hostile reader, not by a lint.

BUT THE MODEL WAS THE LEAST IMPORTANT PART. What made those reviews work was that they were
handed SPECIFIC CLAIMS with file:line pointers and told to verify or refute. A generic "review
this artifact" would not have found the arity bug. A machine cannot write that prompt from
nothing -- so this module's job is to produce the specific, checkable claim, and only then spend
a review on it.

THE CASCADE, and why the cheap layer comes first:

    contradiction detectors   ~0 cost      a row that disagrees with itself
    escalation to a review    ~300k tok    WHY it disagrees, when the detector cannot say

The arity bug was mechanically detectable at the first cell -- load the engine, call it the way
the scorer calls it, see if it raises. Two lines. It ran for twelve cells before a review found
it. The detectors below would have caught it immediately and for free; the review is for the
cases where "something is wrong here" is knowable but "what" is not.

WHAT IT NEVER DOES. Never edits an artifact, never blocks a commit, never fails the conductor --
the same contract as the five audits it sits beside. It surfaces; the operator decides.

WHY THE TRIGGER IS THE HARD PART. Firing on everything makes this a milestone audit with a bigger
bill. Firing on nothing makes it decorative. `artifact_convention_audit` already has a documented
false-positive problem, so a noisy trigger is a known failure mode here rather than a theoretical
one. Every detector below therefore encodes a SELF-CONTRADICTION -- two fields of the same row
that cannot both be right -- rather than a heuristic about what a good result looks like.

Usage:
    python3 scripts/contradiction_escalation.py --recent 12            # detect only
    python3 scripts/contradiction_escalation.py --recent 12 --escalate # spawn reviews on hits
    python3 scripts/contradiction_escalation.py --shard path.jsonl     # per-cell shard
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

REPO = Path(__file__).resolve().parents[1]
REPORT = REPO / "ops" / "contradiction_escalation_report.md"

# A review costs ~200-330k tokens and 10-20 minutes (measured 2026-08-16). Unthrottled, a noisy
# trigger is expensive in a way the scheduled audits are not, so escalation is capped per run and
# the cap is reported rather than silently applied.
MAX_ESCALATIONS = int(os.environ.get("CARNOT_MAX_ESCALATIONS", "3"))


# ---------------------------------------------------------------------------------------------
# Detectors. Each returns a CLAIM STRING when it fires -- the specific, checkable sentence the
# reviewer will be asked to verify -- or None. The string matters as much as the detection: it is
# what turns "review this" into "verify this exact thing", which is the difference between the
# reviews that worked today and a generic pass.
# ---------------------------------------------------------------------------------------------
def _num(row: dict, *names: str) -> Any:
    for n in names:
        v = row.get(n)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v
    return None


def success_flag_with_zero_quality(row: dict) -> str | None:
    """A success flag beside a zero quality metric. THE detector that would have caught the bug.

    `induce_ok=True` means "loadable module defining the required symbols". `cell_recall=0.0`
    means it predicted nothing. Both true at once is not a weak result -- it is a row saying the
    artifact satisfied the interface and failed every prediction, which is the exact signature of
    a STRUCTURAL fault (wrong arity, wrong call convention) rather than a modelling one. On
    2026-08-16 that pair appeared on 4 of 7 cells and was reported for hours as a model weakness;
    the cause was a prompt directive specifying `engine(grid, action)` against a scorer calling
    `engine(grid, action, data)`.
    """
    ok = row.get("induce_ok") or row.get("induction_ok") or row.get("acceptance_gate_passed")
    if not ok:
        return None
    for field in ("cell_recall", "heldout_accuracy", "accuracy", "change_fidelity"):
        v = _num(row, field)
        if v == 0.0:
            flag = "induce_ok" if row.get("induce_ok") else "a pass flag"
            return (
                f"This row reports success ({flag}=True) while `{field}` is exactly 0.0. "
                "Explain how both can be true. Check specifically whether the produced artifact "
                "is STRUCTURALLY incompatible with the scorer -- wrong function arity, wrong "
                "argument order, wrong return type -- such that every scored item raises and is "
                "skipped, leaving the metric computed over an empty set. Quote the scorer's call "
                "site and the artifact's signature."
            )
    return None


def all_cells_identical_outcome(rows: list[dict]) -> str | None:
    """Every cell landing on the same value. Real corpora vary; a constant is usually a pipe."""
    if len(rows) < 4:
        return None
    for field in ("cell_recall", "heldout_accuracy", "induce_ok"):
        vals = [r.get(field) for r in rows if field in r]
        if len(vals) >= 4 and len(set(map(str, vals))) == 1:
            return (
                f"All {len(vals)} cells report the identical `{field}` = {vals[0]!r}. A corpus of "
                "different games producing one constant is more often a broken pipe than a "
                "finding. Determine whether the value is computed at all, or whether every "
                "computation is short-circuiting to the same default."
            )
    return None


def metric_pinned_at_extreme(rows: list[dict]) -> str | None:
    """A metric only ever 0.0 or 1.0 across many cells -- a boolean wearing a float's clothes."""
    for field in ("cell_recall", "heldout_accuracy", "change_fidelity"):
        vals = [_num(r, field) for r in rows]
        vals = [v for v in vals if v is not None]
        if len(vals) >= 6 and set(vals) <= {0.0, 1.0} and len(set(vals)) == 2:
            return (
                f"`{field}` takes only the values 0.0 and 1.0 across {len(vals)} cells, never "
                "anything between. A graded metric that is in practice binary suggests it is "
                "measuring presence/absence of something structural rather than the quality it "
                "names. Identify what makes it saturate."
            )
    return None


def gate_passed_without_the_gated_number(row: dict) -> str | None:
    """A gate declared met while the number it gates on is missing."""
    passed = row.get("acceptance_gate_passed")
    if passed is not True:
        return None
    missing = [
        k
        for k in row
        if (k.endswith("_ready_score") or k.endswith("_score")) and row.get(k) is None
    ]
    if missing:
        return (
            f"`acceptance_gate_passed` is True while {missing[:4]} are null. Establish what the "
            "gate actually evaluated, and whether a null was read as a pass."
        )
    return None


ROW_DETECTORS: tuple[Callable[[dict], str | None], ...] = (
    success_flag_with_zero_quality,
    gate_passed_without_the_gated_number,
)
SET_DETECTORS: tuple[Callable[[list[dict]], str | None], ...] = (
    all_cells_identical_outcome,
    metric_pinned_at_extreme,
)


# ---------------------------------------------------------------------------------------------
# Escalation
# ---------------------------------------------------------------------------------------------
ESCALATION_PROMPT = """\
You are a HOSTILE reviewer. A mechanical detector found a self-contradiction in an experimental
result and cannot classify it. Your job is to find the CAUSE, not to restate the contradiction.

THE SPECIFIC CLAIM TO VERIFY OR REFUTE:
{claim}

Read the actual source. Quote file:line for every assertion. Prefer executing the artifact over
reasoning about it where that is possible.

Consider, in this order, before any explanation that blames the model or the data:
  1. A STRUCTURAL fault -- the artifact is incompatible with the code that scores it (arity,
     argument order, return type, key names), so scoring silently degrades rather than errors.
  2. A METRIC that does not measure what its name says.
  3. A CONFIGURATION constant that means different things to the different things being compared.
  4. Only then, a genuine property of the model or the corpus.

That ordering is not arbitrary. On 2026-08-16 the same contradiction was attributed to model
weakness for several hours; the cause was cause 1, and cause 3 had already produced six earlier
misattributions in the same experiment.

Reply in this exact format:

## VERDICT
<one of: STRUCTURAL_FAULT | METRIC_MISNAMED | CONFIG_ASYMMETRY | GENUINE_RESULT | CANNOT_DETERMINE>

## THE CAUSE
<one paragraph, with file:line>

## EVIDENCE
<quoted code or executed output. Quote identifiers that ARE present in the source.>

## THE FIX
<concrete, or "none needed" if GENUINE_RESULT>
"""


def escalate(claim: str, body: str, agent: str, model: str) -> tuple[str, str]:
    """Spawn one adversarial review. Returns (verdict, report)."""
    sys.path.insert(0, str(REPO / "scripts"))
    from artifact_convention_audit import _call, verify_quoted_evidence  # noqa: E402

    ok, report = _call(agent, model, ESCALATION_PROMPT.format(claim=claim), body)
    if not ok:
        return "CANNOT_DETERMINE", f"reviewer call failed: {report[:200]}"
    verdict = "CANNOT_DETERMINE"
    for v in ("STRUCTURAL_FAULT", "METRIC_MISNAMED", "CONFIG_ASYMMETRY", "GENUINE_RESULT"):
        if v in report.split("## THE CAUSE")[0]:
            verdict = v
            break
    # Reuse the siblings' audit-integrity guard: a reviewer can invent its smoking gun, and a
    # verdict resting on evidence absent from the source must be downgraded rather than acted on.
    invented = verify_quoted_evidence(report, body)
    if invented and verdict != "GENUINE_RESULT":
        return "CANNOT_DETERMINE", (
            f"> Audit-integrity guard: quoted token(s) {invented} do not appear in the source, "
            f"so this verdict was downgraded and must not be acted on.\n\n{report}"
        )
    return verdict, report


def _load_rows(shard: Path | None, recent: int) -> list[tuple[str, dict]]:
    out: list[tuple[str, dict]] = []
    if shard:
        for i, line in enumerate(shard.read_text().splitlines()):
            if line.strip():
                try:
                    out.append((f"{shard.name}#{i}", json.loads(line)))
                except json.JSONDecodeError:
                    continue
        return out
    res = REPO / "results"
    for p in sorted(res.glob("experiment_*.json"), key=lambda q: q.stat().st_mtime)[-recent:]:
        try:
            out.append((p.name, json.loads(p.read_text())))
        except Exception:  # noqa: BLE001
            continue
    return out


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--recent", type=int, default=12)
    ap.add_argument("--shard", help="a per-cell .jsonl shard instead of results/*.json")
    ap.add_argument("--escalate", action="store_true", help="spawn reviews on detections")
    ap.add_argument("--agent-type", default=os.environ.get("AGENT_TYPE_AUDIT", "claude"))
    ap.add_argument("--model-name", default=os.environ.get("AGENT_MODEL_AUDIT", "fable"))
    args = ap.parse_args(argv)

    items = _load_rows(Path(args.shard) if args.shard else None, args.recent)
    if not items:
        print("contradiction-escalation: nothing to inspect")
        return 0

    hits: list[tuple[str, str, dict]] = []
    for name, row in items:
        for det in ROW_DETECTORS:
            claim = det(row)
            if claim:
                hits.append((name, claim, row))
    rows_only = [r for _, r in items]
    for sdet in SET_DETECTORS:
        claim = sdet(rows_only)
        if claim:
            hits.append((f"<{len(rows_only)} cells>", claim, {}))

    print(f"contradiction-escalation: {len(items)} inspected, {len(hits)} contradiction(s)")
    for name, claim, _ in hits:
        print(f"  [{name}] {claim.splitlines()[0][:110]}")

    lines = [
        "# Contradiction escalation",
        "",
        "Cheap detectors for rows that disagree with THEMSELVES, escalated to an adversarial",
        "reviewer when they fire. Never edits, never blocks. Detections are facts; verdicts",
        "downgraded to CANNOT_DETERMINE by the audit-integrity guard rest on evidence the reviewer",
        "could not have read -- do NOT act on those.",
        "",
        f"Inspected {len(items)}, found {len(hits)}.",
        "",
    ]
    if args.escalate and hits:
        if len(hits) > MAX_ESCALATIONS:
            print(
                f"  capping escalations at {MAX_ESCALATIONS} of {len(hits)} "
                f"(~300k tokens each); raise CARNOT_MAX_ESCALATIONS to widen"
            )
            lines.append(
                f"**Only {MAX_ESCALATIONS} of {len(hits)} detections were escalated** "
                "(cost cap). The rest are listed but unreviewed -- silence below is a budget "
                "decision, not a clean bill of health.\n"
            )
        for name, claim, row in hits[:MAX_ESCALATIONS]:
            print(f"  escalating [{name}] ...")
            verdict, report = escalate(
                claim, json.dumps(row, indent=2)[:60000], args.agent_type, args.model_name
            )
            print(f"    -> {verdict}")
            lines += [f"## {name}", "", f"**{verdict}**", "", f"_Claim:_ {claim}", "", report, ""]
    else:
        for name, claim, _ in hits:
            lines += [f"## {name}", "", f"_Claim:_ {claim}", "", "_(not escalated)_", ""]

    REPORT.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {REPORT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
