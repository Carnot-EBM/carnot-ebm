#!/usr/bin/env python3
"""Periodic adversarial AI audit of the QA/reconciliation layer itself:
scripts/adversarial_verify.py, scripts/exclusion_manifest_lint.py,
scripts/in_process_doc_reconcile.py.

This is the sibling of scripts/verifier_authenticity_audit.py -- that audit
polices whether python/carnot/verify/*.py verifiers do what they claim.
Nobody polices the auditor. Every check in this project's fabrication/gate/
reconciliation machinery is itself hand-written pattern-matching code, and
pattern-matching code has exactly the bug class this audit hunts for.

Origin: 2026-07-03 operator question, after a single outer-loop session found
FOUR real bugs in this exact layer in one sitting:
  - in_process_doc_reconcile.py's map_status_label was missing "success" from
    its win-token list, misclassifying ~19% of the ENTIRE historical artifact
    corpus (352+13 of ~4160 artifacts) for an unknown number of months.
  - adversarial_verify.py's _inference_substrate_text() did str(d.get(...))
    on a field that is commonly a {"principle": ..., "value": ...} dict per
    CLAUDE.md's own "Principle-Annotated Artifact Fields" discipline --
    silently defeating substrate recognition on 176 artifacts corpus-wide.
  - adversarial_verify.py's _flips_gate() did a plain substring check
    ("diffusiongemma_met" in verdict) that matched inside the unrelated word
    "meta" (as in meta_tensor, a real PyTorch term) -- no word-boundary
    awareness.
  - adversarial_verify.py had no substrate category for live embedding-only
    LLM calls, so a genuinely fast-but-real 35s embedding-extraction run got
    floored against the 60s full-generation threshold.
None of these were caught by the linter's own 254 pre-existing unit tests
(tests test what the author thought to test, not the shape of real corpus
diversity) or by any existing adversarial audit (out of scope for both
Layer-2 audits that currently exist). The operator's question: "shouldn't
the adversarial agent be catching these?" -- the honest answer was no
adversarial agent has this layer in scope. This script closes that gap.

Two audit granularities, chosen by target-file size:
  - Small files (exclusion_manifest_lint.py, in_process_doc_reconcile.py):
    whole-file audit, mirroring verifier_authenticity_audit.py's pattern.
  - adversarial_verify.py (5600+ lines): function-level chunking. Extracts
    every top-level function via `ast`, filters to functions whose body
    does field extraction / string pattern-matching (the exact bug class
    this audit hunts), and audits each chunk independently with its own
    docstring as context. Auditing the whole file in one LLM call would
    either exceed context or produce a shallow, unfocused review.

The audit produces a structured report at
ops/qa_layer_authenticity_audit_report.md. It does NOT edit any file --
operator decides what to act on. Reuses the same audit-integrity guard
(Layer 1.5) as verifier_authenticity_audit.py: an LLM hostile reviewer can
hallucinate its smoking gun, so any flagged verdict whose quoted
high-specificity evidence does not literally appear in the source is
auto-downgraded to CANNOT_DETERMINE rather than acted on.

Usage:
    python scripts/qa_layer_authenticity_audit.py [--model gemini|claude|codex]
                                                   [--file FILE]
                                                   [--limit N]

Designed to be called from the conductor's milestone-close path (see
scripts/research_conductor.py:_run_operational_retrospective's caller),
alongside the existing verifier and landing-page audits.
"""

from __future__ import annotations

import argparse
import ast
import datetime
import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPORT_PATH = PROJECT_ROOT / "ops" / "qa_layer_authenticity_audit_report.md"

# Whole-file targets: small enough to audit in one shot.
WHOLE_FILE_TARGETS = (
    PROJECT_ROOT / "scripts" / "exclusion_manifest_lint.py",
    PROJECT_ROOT / "scripts" / "in_process_doc_reconcile.py",
)

# Function-chunked target: too large for a whole-file audit.
CHUNKED_FILE_TARGETS = (PROJECT_ROOT / "scripts" / "adversarial_verify.py",)

# A function body is audit-worthy (does field extraction / pattern-matching --
# the exact bug class this audit hunts) if it contains any of these markers.
# Excludes pure scaffolding (argparse setup, report string-building, Flag
# dataclass definitions) that has no field-shape or substring-boundary risk.
RISKY_BODY_MARKERS = (
    ".get(",
    " in ",
    ".lower()",
    ".upper()",
    "re.search(",
    "re.match(",
    "re.compile(",
    ".startswith(",
    ".endswith(",
    "isinstance(",
)

PER_CHUNK_PROMPT = """\
You are a hostile software reviewer auditing a piece of QA/fabrication-detection
code from the Carnot project's adversarial_verify.py -- the mechanical linter
that runs on every research-experiment artifact to catch fabricated or
implausible results (fake AUROC numbers, impossibly-short compute durations,
gamed statistical tests, etc).

This code is CRITICAL infrastructure: a bug here either lets fabrication
through (a false negative -- dangerous, the linter's whole purpose fails
silently) or falsely quarantines honest work (a false positive -- costly,
wastes investigation time and can mislead the project into thinking real
progress is suspect). A single outer-loop session on 2026-07-03 found FOUR
real bugs of this kind in one sitting:

1. A field-shape assumption bug: code did `str(d.get("some_field"))` assuming
   the field is always a bare string/number, but the project's own
   "Principle-Annotated Artifact Fields" convention (documented in CLAUDE.md)
   allows ANY field to be written as `{"principle": "...", "value": ...}`.
   `str()` on that dict produces a Python repr matching nothing -- silent
   failure across 176 real artifacts.

2. A substring-boundary bug: `"diffusiongemma_met" in verdict.lower()` matched
   inside the unrelated word "meta" (as in "meta_tensor", a real PyTorch
   term) purely by coincidence -- no word-boundary check.

3. A substrate-taxonomy gap: a real, honest, fast-but-genuine compute pattern
   (embedding-only LLM calls, much cheaper than full generation) had no
   matching category, so it fell through to a floor calibrated for a
   completely different (much more expensive) workload.

Your job: find more bugs like these in the function below. Answer THIS
structured set of questions. Do not soften the answers. Do not rationalize.

1. **Field extraction assumptions.** For every `d.get("field_name")` or
   similar dict-field read: does the code assume a specific TYPE (bare
   string, bare number, bare bool) without handling the case where the
   field might be a `{"principle": ..., "value": ...}` wrapped dict, a list,
   or None? Quote the specific line.

2. **String/substring matching without boundaries.** For every `in`,
   `.startswith(`, `.endswith(`, or `re.search`/`re.match` against free-text
   fields (honest_verdict, inference_substrate, docstrings, etc.): could the
   pattern match INSIDE an unrelated, longer word or phrase that happens to
   contain it as a substring? Try to construct a concrete counterexample
   string that would falsely match (or falsely fail to match).

3. **Negation / context blindness.** Does any check that scans free text for
   a forbidden/flagged phrase fail to distinguish "the artifact DOES X" from
   "the artifact explicitly did NOT do X" / "the artifact correctly avoided
   X" / "this check exists to detect X"? (E.g. would a verdict saying
   "blocked_X_not_attempted" incorrectly trigger a check meant to catch
   artifacts that DID X?)

4. **Boundary / off-by-one errors.** For any numeric threshold, comparison,
   or floor: is the comparison operator correct (`>` vs `>=`, `<` vs `<=`)?
   Does it handle the exact-equal-to-threshold case the way the docstring
   implies it should?

5. **Does the implementation match what the function's own docstring or name
   claims it detects?** Quote the claim, then say whether the actual logic
   is narrower, broader, or different from that claim.

6. **Construct one concrete, realistic artifact fragment** (a plausible
   honest_verdict string, or a plausible field dict) that would be
   MIS-classified by this function -- either a false positive (flags honest
   work) or a false negative (misses something it should catch). If you
   cannot construct one, say so explicitly.

Output format -- exactly this structure (no preamble, no postscript):

```
## VERDICT
<one of: CLEAN | MINOR_RISK | REAL_BUG | CANNOT_DETERMINE>

## CLAIM
<what the function's docstring/name claims to do, 1 sentence>

## FINDINGS
<numbered list of concrete issues found per the 6 questions above, or "none found">

## COUNTEREXAMPLE
<the concrete mis-classifying input you constructed, or "none constructed">

## RECOMMENDATION
<one of: KEEP | ADD_WORD_BOUNDARY | ADD_FIELD_UNWRAP | ADD_TEST_CASE | NEEDS_REDESIGN>

## RATIONALE
<2-3 sentences>
```

Be hostile. If you find no problems, say CLEAN and move on -- do not invent
issues to seem thorough.
"""

PER_FILE_PROMPT = """\
You are a hostile software reviewer auditing a QA/reconciliation script from
the Carnot project -- code that classifies, gates, or reconciles research
artifacts and roadmap tasks. A single outer-loop session on 2026-07-03 found
a real bug of exactly this shape: in_process_doc_reconcile.py's
map_status_label was missing "success" from its list of terminal-success
tokens, silently misclassifying ~19% of the entire historical artifact
corpus (352+13 of ~4160 artifacts, going back many months) as something
other than Complete.

Your job: review the WHOLE FILE below for the same class of bug. Answer:

1. **Classification/token lists.** For every list of tokens, prefixes, or
   markers used to classify a status/verdict/pattern: is it plausibly
   INCOMPLETE (missing an obvious synonym or common real-world phrasing)?
   Cross-check against the project's documented "Verdict Terminal-Prefix
   Discipline" (CLAUDE.md): terminal verdicts start with complete:/complete_/
   success:/success_/passed:/passed_/shipped:/shipped_. Does every one of
   those 8 prefix forms have matching recognition logic?

2. **Substring matching without boundaries.** Same question as for
   adversarial_verify.py -- could a pattern match inside an unrelated
   longer word?

3. **Field-shape assumptions.** Does this file assume fields are bare
   strings/numbers when the project's "Principle-Annotated Artifact Fields"
   convention allows `{"principle": ..., "value": ...}` wrapping?

4. **Negation / context blindness.** Does any forbidden-pattern scan fail
   to distinguish "did X" from "explicitly did not do X" / "correctly
   avoided X"?

5. **Construct one concrete, realistic counterexample** (a plausible
   verdict string or roadmap task fragment) that this file would
   mis-classify.

Output format -- exactly this structure (no preamble, no postscript):

```
## VERDICT
<one of: CLEAN | MINOR_RISK | REAL_BUG | CANNOT_DETERMINE>

## FINDINGS
<numbered list of concrete issues, or "none found">

## COUNTEREXAMPLE
<the concrete mis-classifying input you constructed, or "none constructed">

## RECOMMENDATION
<one of: KEEP | ADD_WORD_BOUNDARY | ADD_FIELD_UNWRAP | ADD_TOKEN | ADD_TEST_CASE | NEEDS_REDESIGN>

## RATIONALE
<2-3 sentences>
```

Be hostile. If you find no problems, say CLEAN and move on -- do not invent
issues to seem thorough.
"""


@dataclass
class Chunk:
    label: str  # e.g. "adversarial_verify.py::_flips_gate"
    body: str
    source_file: Path


def extract_risky_functions(path: Path) -> list[Chunk]:
    """Extract top-level function defs whose body does field extraction or
    string pattern-matching -- the exact bug class this audit hunts. Skips
    pure scaffolding (argparse, report formatting, dataclasses) with no such
    risk."""
    try:
        source = path.read_text()
    except Exception:
        return []
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []
    lines = source.splitlines()
    chunks: list[Chunk] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if getattr(node, "col_offset", 0) != 0:
            continue  # only top-level functions, not nested/methods
        start = node.lineno - 1
        end = getattr(node, "end_lineno", node.lineno)
        body = "\n".join(lines[start:end])
        if not any(marker in body for marker in RISKY_BODY_MARKERS):
            continue
        if len(body) < 40:
            continue
        chunks.append(Chunk(label=f"{path.name}::{node.name}", body=body, source_file=path))
    return chunks


def call_gemini(prompt: str, body: str, model: str = "gemini-3.1-pro-preview") -> tuple[bool, str]:
    try:
        full = f"{prompt}\n\n---\nCODE:\n\n{body}"
        proc = subprocess.run(
            ["gemini", "--model", model, "--yolo", "-p", full],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            cwd=PROJECT_ROOT,
        )
        if proc.returncode != 0:
            return False, f"gemini exit {proc.returncode}: {proc.stderr[:200]}"
        return True, proc.stdout
    except Exception as exc:
        return False, str(exc)


def call_claude(prompt: str, body: str, model: str = "claude-opus-4-8") -> tuple[bool, str]:
    try:
        full = f"{prompt}\n\n---\nCODE:\n\n{body}"
        proc = subprocess.run(
            ["claude", "--model", model, "--effort", "max", "--print", full],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
            cwd=PROJECT_ROOT,
        )
        if proc.returncode != 0:
            return False, f"claude exit {proc.returncode}: {proc.stderr[:200]}"
        return True, proc.stdout
    except Exception as exc:
        return False, str(exc)


def call_codex(prompt: str, body: str, model: str = "gpt-5.5") -> tuple[bool, str]:
    """Codex (gpt-5.5) hostile reviewer -- quota-conserve path, mirrors verifier_authenticity_audit.py."""
    try:
        full = f"{prompt}\n\n---\nCODE:\n\n{body}"
        proc = subprocess.run(
            [
                "codex",
                "exec",
                "--dangerously-bypass-approvals-and-sandbox",
                "--color",
                "never",
                "--model",
                model,
                "--cd",
                str(PROJECT_ROOT),
                "--ephemeral",
                "-",
            ],
            input=full,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
            cwd=PROJECT_ROOT,
        )
        if proc.returncode != 0:
            return False, f"codex exit {proc.returncode}: {proc.stderr[:200]}"
        return True, proc.stdout
    except Exception as exc:
        return False, str(exc)


def call_model(model_kind: str, model_name: str | None, prompt: str, body: str) -> tuple[bool, str]:
    if model_kind == "gemini":
        return call_gemini(prompt, body, model=model_name or "gemini-3.1-pro-preview")
    if model_kind == "codex":
        return call_codex(prompt, body, model=model_name or "gpt-5.5")
    return call_claude(prompt, body, model=model_name or "sonnet")


def parse_verdict(report: str) -> str:
    m = re.search(r"##\s*VERDICT\s*\n\s*(\S+)", report)
    return m.group(1).strip() if m else "UNKNOWN"


def verify_quoted_evidence(report: str, body: str) -> tuple[list[str], list[str]]:
    """Audit-integrity guard (Layer 1.5) -- identical logic to
    verifier_authenticity_audit.py's guard of the same name. An LLM hostile
    reviewer can hallucinate its smoking gun (a quoted line, a fabricated
    counterexample string, a path that doesn't exist). We can't fact-check
    prose, but we CAN fact-check high-specificity quoted spans (backtick
    code spans that look like a file path, a line of code, or contain a
    distinctive identifier) against the actual source. Missing evidence ->
    the flagged verdict is auto-downgraded rather than acted on.
    """
    norm_body = re.sub(r"\s+", "", body)

    def is_present(core: str) -> bool:
        if re.sub(r"\s+", "", core) in norm_body:
            return True
        toks = re.findall(r"[A-Za-z0-9_./-]{6,}", core)
        return any(re.sub(r"\s+", "", t) in norm_body for t in toks)

    high: list[str] = []
    missing: list[str] = []
    for span in re.findall(r"`([^`]+)`", report):
        core = span.strip()
        if len(core) < 6:
            continue
        is_high_specificity = bool(
            re.search(r"\.(json|py|ya?ml|txt|md)\b", core)
            or "/" in core
            or re.search(
                r"\b(d\.get|re\.search|re\.match|\.lower\(\)|\.startswith|\.endswith|"
                r"np\.random|numpy\.random|time\.sleep|torch\.load)\b",
                core,
            )
        )
        if not is_high_specificity:
            continue
        high.append(core)
        if not is_present(core):
            missing.append(core)
    return high, missing


def _run_one(
    label: str,
    body: str,
    prompt: str,
    args: argparse.Namespace,
    out: list[str],
    counts: dict[str, int],
    flagged: list[tuple[str, str]],
    integrity_voids: list[tuple[str, str, list[str]]],
) -> None:
    flagged_verdicts = {"REAL_BUG", "NEEDS_REDESIGN"}
    ok, report = call_model(args.model, args.model_name, prompt, body)
    if not ok:
        out.append(f"## {label}\n\n(audit call failed: {report[:200]})\n")
        counts["UNKNOWN"] = counts.get("UNKNOWN", 0) + 1
        return
    verdict = parse_verdict(report)
    guard_note = ""
    if verdict in flagged_verdicts:
        _high, missing = verify_quoted_evidence(report, body)
        if missing:
            integrity_voids.append((label, verdict, missing[:6]))
            guard_note = (
                "\n> **AUDIT-INTEGRITY GUARD (Layer 1.5) — VERDICT AUTO-DOWNGRADED.** "
                f"The `{verdict}` verdict cited high-specificity evidence (code spans / "
                "file paths / distinctive identifiers) that does NOT appear in the source "
                "chunk (checked literally + by distinctive sub-token). This is the auditor "
                "hallucinating its smoking gun. Verdict downgraded to `CANNOT_DETERMINE` "
                "and removed from the action list; DO NOT act on this basis. Absent "
                "evidence: " + "; ".join(f"`{m}`" for m in missing[:6]) + "\n"
            )
            print(
                f"    [integrity-guard] VOIDED {label}: {verdict} cited "
                f"{len(missing)} absent evidence string(s)",
                file=sys.stderr,
            )
            verdict = "CANNOT_DETERMINE"
    counts[verdict] = counts.get(verdict, 0) + 1
    if verdict in flagged_verdicts:
        flagged.append((label, verdict))
    out.append(f"## {label}\n")
    out.append(f"**Verdict:** `{verdict}`\n")
    out.append(report.strip())
    if guard_note:
        out.append(guard_note)
    out.append("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="claude",  # matches verifier_authenticity_audit.py's default; gemini is never
        # the default per the 2026-06-10 global-stall directive.
        choices=["gemini", "claude", "codex"],
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--file",
        default=None,
        help="Audit a single file (chunked if it's adversarial_verify.py, whole-file otherwise)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N chunks/files total (for time-bounded sampling)",
    )
    args = parser.parse_args()

    units: list[tuple[str, str, str]] = []  # (label, body, prompt)

    if args.file:
        p = Path(args.file)
        if p in CHUNKED_FILE_TARGETS:
            for chunk in extract_risky_functions(p):
                units.append((chunk.label, chunk.body, PER_CHUNK_PROMPT))
        else:
            try:
                units.append((p.name, p.read_text(), PER_FILE_PROMPT))
            except Exception:
                pass
        rotated = False
    else:
        for p in WHOLE_FILE_TARGETS:
            try:
                units.append((p.name, p.read_text(), PER_FILE_PROMPT))
            except Exception:
                continue
        for p in CHUNKED_FILE_TARGETS:
            for chunk in extract_risky_functions(p):
                units.append((chunk.label, chunk.body, PER_CHUNK_PROMPT))
        rotated = True

    # Rotation state: adversarial_verify.py alone has 150+ risky-function chunks --
    # far more than a single milestone-close pass (--limit ~20-25) can cover. Without
    # rotation, a fixed head-slice would always re-audit the same first N units and
    # NEVER reach the rest (the exact limitation verifier_authenticity_audit.py has,
    # unaddressed, for its own --limit 20). Persist an offset so successive full-corpus
    # runs advance through the whole list over time; --file runs (single-target,
    # rotated=False) don't touch rotation state.
    if args.limit > 0 and rotated and units:
        state_path = PROJECT_ROOT / "ops" / ".qa_layer_audit_rotation.json"
        offset = 0
        try:
            offset = int(json.loads(state_path.read_text()).get("offset", 0))
        except Exception:
            offset = 0
        offset = offset % len(units)
        rotated_units = units[offset:] + units[:offset]
        units = rotated_units[: args.limit]
        try:
            state_path.write_text(
                json.dumps({"offset": (offset + args.limit) % len(rotated_units)})
            )
        except Exception:
            pass
    elif args.limit > 0:
        units = units[: args.limit]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()

    out = [
        f"<!-- generated by scripts/qa_layer_authenticity_audit.py — {today} -->",
        f"<!-- per CLAUDE.md 'QA-Layer Authenticity Discipline' — advisory only -->",
        f"",
        f"# qa_layer_authenticity_audit_report — {today}",
        f"",
        f"Scanned {len(units)} unit(s) (whole-files + function-chunks) with {args.model} as "
        f"the hostile reviewer. Targets: {', '.join(p.name for p in WHOLE_FILE_TARGETS)} "
        f"(whole-file), {', '.join(p.name for p in CHUNKED_FILE_TARGETS)} (function-chunked).",
        f"",
    ]

    counts: dict[str, int] = {
        "CLEAN": 0,
        "MINOR_RISK": 0,
        "REAL_BUG": 0,
        "CANNOT_DETERMINE": 0,
        "NEEDS_REDESIGN": 0,
        "UNKNOWN": 0,
    }
    flagged: list[tuple[str, str]] = []
    integrity_voids: list[tuple[str, str, list[str]]] = []

    for i, (label, body, prompt) in enumerate(units, 1):
        print(f"[{i}/{len(units)}] {label}", file=sys.stderr)
        _run_one(label, body, prompt, args, out, counts, flagged, integrity_voids)

    summary = [
        "## Summary",
        "",
        "| Verdict | Count |",
        "|---|---|",
    ]
    for v, n in counts.items():
        summary.append(f"| `{v}` | {n} |")
    if flagged:
        summary.append("")
        summary.append("### FLAGGED — operator action recommended")
        for label, verdict in flagged:
            summary.append(f"- `{label}` — **{verdict}**")
    if integrity_voids:
        summary.append("")
        summary.append(
            "### AUDIT-INTEGRITY GUARD — flags voided (auditor hallucinated its evidence)"
        )
        summary.append(
            "These verdicts were FLAGGED by the LLM reviewer but cited concrete code/path "
            "strings that do NOT exist in the source chunk. Auto-downgraded to "
            "`CANNOT_DETERMINE`; **do NOT act on them.** They indicate the audit RUN was "
            "partly unreliable, not that the code is buggy."
        )
        for label, verdict, missing in integrity_voids:
            ev = "; ".join(f"`{m}`" for m in missing) if missing else "(none captured)"
            summary.append(f"- `{label}` — was **{verdict}**; absent evidence: {ev}")
    summary.append("")
    summary.append("---")
    summary.append("")
    out = out[:5] + summary + out[5:]

    REPORT_PATH.write_text("\n".join(out))
    print(f"audit complete — report at {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  scanned: {len(units)} unit(s)")
    print(f"  flagged: {len(flagged)}")
    for label, verdict in flagged:
        print(f"    {label}: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
