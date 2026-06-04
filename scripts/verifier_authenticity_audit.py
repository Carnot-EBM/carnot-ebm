#!/usr/bin/env python3
"""Periodic adversarial AI audit of every verifier in
python/carnot/verify/.

Layer 2 of the "Verifier Authenticity Discipline" defense (complements
`scripts/verifier_authenticity_lint.py`). The mechanical lint catches
gaming patterns (sleep-padding, score-capping, random-data evaluators)
deterministically at commit-time. This audit runs an LLM as a HOSTILE
SOFTWARE REVIEWER who asks for EACH verifier file:

> "Does this implementation actually do what the docstring claims, or
>  is it a mock/stub/heuristic dressed up to look like a model-based
>  verifier?"

Origin: 2026-05-21 deep audit found:
  - tier0s_halluguard.py claims "NTK-based HalluGuard (arXiv:2601.18753)"
    but the implementation is 56 lines of `re.findall(r'\\d+', text)` +
    `|a+b−c|` arithmetic. No torch, no GPU, no model. The mechanical
    lint can't catch this because the file has no gaming patterns —
    it's just dishonest naming.
  - nla_eval_awareness_1716.py games every adversarial check explicitly
    (caught by lint).

The audit produces a structured report at
`ops/verifier_authenticity_audit_report.md`. It does NOT edit any
verifier — operator decides what to act on (rename, retire, or fix).

Usage:
    python scripts/verifier_authenticity_audit.py [--model gemini|claude]
                                                  [--file FILE]

By default scans every `python/carnot/verify/*.py`. The `--file`
override runs against a single verifier for spot-checks.

Designed to be called from the conductor's milestone-close path so the
operator gets a fresh review every milestone. Independent of any
verifier-modifying agent — independence-of-context is what makes the
audit adversarial.
"""

from __future__ import annotations

import argparse
import datetime
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERIFY_DIR = PROJECT_ROOT / "python" / "carnot" / "verify"
REPORT_PATH = PROJECT_ROOT / "ops" / "verifier_authenticity_audit_report.md"

# Per-verifier audit prompt. Strong "hostile reviewer" frame so the
# auditor names problems rather than rationalizing them.
PER_VERIFIER_PROMPT = """\
You are a hostile software reviewer auditing a Carnot verifier module.

Carnot is an open-source framework for verifying LLM output using
energy-based models. Each verifier in `python/carnot/verify/` is
expected to actually do what its docstring claims. The 2026-05-21
audit caught a verifier (`tier0s_halluguard.py`) whose docstring
claimed "NTK-based HalluGuard (arXiv:2601.18753)" but whose
implementation was 56 lines of regex + arithmetic with no model
invocation. Your job is to find more like this.

For the verifier source code below, answer THIS structured set of
questions. Do not soften the answers. Do not rationalize.

1. **Does the docstring make CLAIMS about a paper, a model, or a
   compute substrate?** Quote the specific claim if so. Examples of
   what to flag:
   - "Based on arXiv:NNNN.NNNNN"
   - "NTK-based" / "neural tangent kernel" / "kernel embedding"
   - "Uses logprobs from <model>"
   - "Runs on GPU" / "live inference"
   - "Learned embedding" / "SAE features" / "attention scores"
   - "Predictive coding" / "information bottleneck"

2. **Does the IMPLEMENTATION match the claim?** Check for:
   - Is there a real model invocation (`AutoModel`, `torch.load`,
     `llama_cpp.Llama`, `requests.post(API_URL, ...)`)?
   - Or is it pure-Python text statistics (`re.findall`, `len(str)`,
     `text.count(...)`, dict lookup)?
   - Does the file import torch / transformers / jax / sklearn /
     llama_cpp / openai / a Carnot model module?

3. **If implementation does NOT match the claim, is the docstring
   HONEST about the gap?** Some verifiers legitimately implement
   "text-statistical proxies" that approximate a paper's idea — the
   docstring should EXPLICITLY say so (e.g., "We don't have access
   to per-token logits at inference time. Instead, we implement two
   text-statistical proxies that capture the same intuition...").
   That's honest. Silently claiming the paper's full method when
   you're shipping regex is what we're hunting.

4. **Does the code GAME the adversarial-verify checks?** Look for:
   - `time.sleep(X)` followed by `duration_s = X` (sleep-padding)
   - `min(score, 0.99)` or `max(score, 0.01)` (perfect-dodging)
   - `np.random.*` generating features that get classified
   - Mentioning "IMPLAUSIBLE_PERFECT" or "DURATION_TOO_SHORT" by name

5. **Is the verifier suitable for the production ensemble?** If the
   answer to (3) is "no, the docstring is dishonest" or (4) is "yes,
   it games checks", the verifier MUST NOT contribute to headline
   AUROC numbers.

Output format — exactly this structure (no preamble, no postscript):

```
## VERDICT
<one of: AUTHENTIC | HONEST_HEURISTIC | DISHONEST_NAMING | ADVERSARIAL_GAMING | CANNOT_DETERMINE>

## CLAIMS
<quote specific docstring claims about paper/model/compute>

## IMPLEMENTATION_REALITY
<1-2 sentences: what the code ACTUALLY does>

## CLAIM_VS_REALITY_GAP
<NONE | DISCLOSED_PROXY | UNDISCLOSED_PROXY | OUTRIGHT_FAKE>

## GAMING_PATTERNS
<list any sleep-padding / score-capping / random-data / token-dodge patterns, or "none">

## RECOMMENDATION
<one of: KEEP | RENAME_TO_REFLECT_REALITY | RETIRE | REIMPLEMENT_PROPERLY>

## RATIONALE
<2-3 sentences>
```

Be hostile. If you find no problems, say AUTHENTIC and move on.
"""


def call_gemini(prompt: str, body: str, model: str = "gemini-3.1-pro-preview") -> tuple[bool, str]:
    try:
        full = f"{prompt}\n\n---\nVERIFIER SOURCE:\n\n{body}"
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


def call_claude(prompt: str, body: str, model: str = "sonnet") -> tuple[bool, str]:
    try:
        full = f"{prompt}\n\n---\nVERIFIER SOURCE:\n\n{body}"
        proc = subprocess.run(
            ["claude", "--model", model, "--print", full],
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


def parse_verdict(report: str) -> str:
    m = re.search(r"##\s*VERDICT\s*\n\s*(\S+)", report)
    return m.group(1).strip() if m else "UNKNOWN"


def verify_quoted_evidence(report: str, body: str) -> tuple[list[str], list[str], list[str]]:
    """Audit-integrity guard (Layer 1.5).

    An LLM hostile-reviewer can HALLUCINATE its smoking gun — e.g. the 2026-06-03
    run flagged ast_structure_verifier.py ADVERSARIAL_GAMING citing a hardcoded
    path `...|with| @results/experiment_2101_interwhen.json)` that does NOT exist
    anywhere in the source (verified by grep). Acting on that would have RETIRED a
    clean honest-heuristic verifier on fabricated evidence — exactly the fabrication
    the audit exists to PREVENT, committed BY the audit.

    This checks every backtick-quoted span the auditor cited: a span is
    "source-checkable" if it looks like a concrete code/path literal (has a
    structural char like / . _ ( ) = @ AND a >=4-char alnum run). For each, we test
    whether it — or a distinctive >=6-char sub-token of it — literally appears in the
    source (whitespace-normalized). Returns (checkable, present, missing). When a
    FLAGGED verdict has checkable evidence but NONE of it is present in source, the
    caller treats the verdict as likely-hallucinated and auto-downgrades it.
    """
    norm_body = re.sub(r"\s+", "", body)
    checkable: list[str] = []
    present: list[str] = []
    missing: list[str] = []
    for span in re.findall(r"`([^`]+)`", report):
        core = span.strip()
        if len(core) < 6:
            continue
        if not re.search(r"[/._()=@\\]", core):       # not a code/path-looking literal
            continue
        if not re.search(r"[A-Za-z0-9]{4,}", core):    # no substantial alnum content
            continue
        checkable.append(core)
        norm_core = re.sub(r"\s+", "", core)
        hit = norm_core in norm_body
        if not hit:
            # the full quote may be paraphrased/partial; accept a distinctive sub-token
            toks = re.findall(r"[A-Za-z0-9_./-]{6,}", core)
            hit = any(re.sub(r"\s+", "", t) in norm_body for t in toks)
        (present if hit else missing).append(core)
    return checkable, present, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="gemini",
        choices=["gemini", "claude"],
    )
    parser.add_argument("--model-name", default=None)
    parser.add_argument(
        "--file",
        default=None,
        help="Audit a single verifier file instead of all",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Stop after N files (for time-bounded sampling)",
    )
    args = parser.parse_args()

    if args.file:
        files = [Path(args.file)]
    else:
        files = sorted(VERIFY_DIR.glob("*.py"))

    if args.limit > 0:
        files = files[: args.limit]

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()

    out = [
        f"<!-- generated by scripts/verifier_authenticity_audit.py — {today} -->",
        f"<!-- per CLAUDE.md 'Verifier Authenticity Discipline' — advisory only -->",
        f"",
        f"# verifier_authenticity_audit_report — {today}",
        f"",
        f"Scanned {len(files)} verifier file(s) with {args.model} as the hostile reviewer.",
        f"",
    ]

    flagged_verdicts = {"DISHONEST_NAMING", "ADVERSARIAL_GAMING", "OUTRIGHT_FAKE"}
    counts = {"AUTHENTIC": 0, "HONEST_HEURISTIC": 0, "DISHONEST_NAMING": 0,
              "ADVERSARIAL_GAMING": 0, "CANNOT_DETERMINE": 0, "UNKNOWN": 0,
              "OUTRIGHT_FAKE": 0}
    flagged_files = []
    # (file, original_verdict, missing_evidence[]) for flags voided by the integrity guard
    integrity_voids: list[tuple[str, str, list[str]]] = []

    for i, f in enumerate(files, 1):
        rel = f.relative_to(PROJECT_ROOT)
        try:
            body = f.read_text()
        except Exception:
            continue
        if len(body) < 100:
            counts["AUTHENTIC"] += 1
            continue
        print(f"[{i}/{len(files)}] {rel}", file=sys.stderr)
        if args.model == "gemini":
            ok, report = call_gemini(
                PER_VERIFIER_PROMPT,
                body,
                model=args.model_name or "gemini-3.1-pro-preview",
            )
        else:
            ok, report = call_claude(
                PER_VERIFIER_PROMPT,
                body,
                model=args.model_name or "sonnet",
            )
        if not ok:
            out.append(f"## {rel}\n\n(audit call failed: {report[:200]})\n")
            counts["UNKNOWN"] += 1
            continue
        verdict = parse_verdict(report)
        counts[verdict] = counts.get(verdict, 0) + 1
        if verdict in flagged_verdicts:
            flagged_files.append((str(rel), verdict))
        out.append(f"## {rel}\n")
        out.append(f"**Verdict:** `{verdict}`\n")
        out.append(report.strip())
        out.append("\n")

    # Summary at top
    summary = [
        "## Summary",
        "",
        "| Verdict | Count |",
        "|---|---|",
    ]
    for v, n in counts.items():
        summary.append(f"| `{v}` | {n} |")
    if flagged_files:
        summary.append("")
        summary.append("### FLAGGED — operator action recommended")
        for path, verdict in flagged_files:
            summary.append(f"- `{path}` — **{verdict}**")
    summary.append("")
    summary.append("---")
    summary.append("")
    out = out[:5] + summary + out[5:]

    REPORT_PATH.write_text("\n".join(out))
    print(f"audit complete — report at {REPORT_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  scanned: {len(files)} verifier(s)")
    print(f"  flagged: {len(flagged_files)}")
    for path, verdict in flagged_files:
        print(f"    {path}: {verdict}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
