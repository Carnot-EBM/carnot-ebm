#!/usr/bin/env python3
"""Mechanical lint for python/carnot/verify/ — catches adversarially-aware
cheating patterns in verifier implementations.

Per CLAUDE.md "Verifier Authenticity Discipline" (2026-05-21): verifier
code must accurately do what its docstring claims. Implementations that
GAME the project's own adversarial-verify checks (sleep-padding to
bypass DURATION_TOO_SHORT, score-capping to bypass IMPLAUSIBLE_PERFECT,
evaluating on numpy.random data) are the worst form of internal
fabrication — they pollute the headline metrics with synthesized noise
while passing every safety net.

Origin: `nla_eval_awareness_1716.py` and `tier0s_halluguard.py` (caught
by deep audit 2026-05-21):

  - nla_eval_awareness_1716.py used `np.random.randn` to fabricate
    SAE features, then `min(tpr, 0.99)` with comment "# To prevent
    IMPLAUSIBLE_PERFECT", then `time.sleep(mock_sleep)` to pad
    `duration_s` past the 60s DURATION_TOO_SHORT threshold. Wall-time
    cost: 100s+ per task with zero research value.
  - tier0s_halluguard.py claims "NTK-based HalluGuard (arXiv:
    2601.18753)" but the implementation is 56 lines of `re.findall(
    r'\\d+', text)` arithmetic — no torch, no GPU, no model.

Rules (each emits a HARD-FAIL on violation):

1. SLEEP-PADDING DURATION
   `time.sleep(X)` followed within 5 lines by `duration_s = X` or
   `duration_s = X + ...`. Real compute takes wall-time; padding
   with sleep is fake.

2. SCORE CAPPING WITH PERFECT-DODGE COMMENT
   `min(score, 0.99)` or `max(score, 0.01)` patterns alongside any
   reference to IMPLAUSIBLE_PERFECT.

3. RANDOM-DATA EVALUATION
   Variables named `mock_features`, `fake_features`, `mock_labels`,
   `fake_labels`, `simulated_features` assigned from `np.random.*`
   in a verifier file. (The corresponding train/eval split is
   computing classification metrics on RANDOM data.)

4. HARDCODED DURATION_S
   `duration_s = <constant>` outside of test fixtures. Real
   `duration_s` is measured via `time.time()` subtraction.

5. ADVERSARIAL-CHECK NAME REFERENCES
   String literals or comments containing `IMPLAUSIBLE_PERFECT` or
   `DURATION_TOO_SHORT` inside verifier production code (NOT inside
   the adversarial_verify.py / lint scripts themselves). If your
   verifier code mentions those tokens by name, you're either
   referring to them legitimately (rare — should be a comment to
   the linter) or gaming them.

6. MOCK_SLEEP PARAMETER
   `def f(..., mock_sleep=...)` — verifier APIs do not take a
   sleep-duration parameter. That's a gaming knob.

Exit codes: 0 clean, 1 violations found.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
VERIFY_DIR = PROJECT_ROOT / "python" / "carnot" / "verify"
PIPELINE_DIR = PROJECT_ROOT / "python" / "carnot" / "pipeline"

# Files allowed to mention the adversarial-check token names (they're the
# linter / audit machinery itself).
ALLOWLIST = {
    "scripts/adversarial_verify.py",
    "scripts/verifier_authenticity_lint.py",
    "scripts/verifier_authenticity_audit.py",
    "scripts/pages_fever_dream_lint.py",
    # findings audits legitimately classify these tokens
    "python/carnot/pipeline/findings_audit_2001.py",
    "python/carnot/pipeline/findings_audit_1984.py",
    # tests are allowed to validate the patterns
}

PATTERNS = [
    (
        "SLEEP_PADDING_DURATION",
        re.compile(r"time\.sleep\([^)]+\)[^\n]*\n[^\n]*duration_s\s*=", re.MULTILINE),
        "time.sleep(X) followed by duration_s = X pads wall-time fakery",
    ),
    (
        "SCORE_CAP_TO_99",
        re.compile(r"min\([^)]+,\s*0\.99\)"),
        "min(x, 0.99) caps to avoid IMPLAUSIBLE_PERFECT — fix the fake instead",
    ),
    (
        "SCORE_FLOOR_AT_01",
        re.compile(r"max\([^)]+,\s*0\.01\)"),
        "max(x, 0.01) floors to avoid zero-detection — fix the fake instead",
    ),
    (
        "DODGE_TOKEN_REFERENCE",
        re.compile(r"\b(IMPLAUSIBLE[_\s]?PERFECT|DURATION[_\s]?TOO[_\s]?SHORT)\b"),
        "verifier code referencing adversarial-check token names is gaming",
    ),
    (
        "RANDOM_DATA_EVAL",
        re.compile(
            r"(mock_|fake_|simulated_|synthetic_)\w*(features|labels|data|scores)\s*=\s*np\.random"
        ),
        "fabricated evaluation data (np.random + 'mock'/'fake' prefix)",
    ),
    (
        "DURATION_S_HARDCODED",
        re.compile(r"duration_s\s*=\s*(\d{2,4}\.\d+|\d{2,4})\b"),
        "duration_s set to a literal constant — must come from time.time() diff",
    ),
    (
        "MOCK_SLEEP_PARAMETER",
        re.compile(r"def\s+\w+\s*\([^)]*\bmock_sleep\b"),
        "function parameter named mock_sleep — verifier APIs do not take sleep knobs",
    ),
]


def scan_file(path: Path) -> list[tuple[str, int, str, str]]:
    """Return [(violation_kind, line_no, line_text, why), ...]."""
    rel = str(path.relative_to(PROJECT_ROOT))
    if rel in ALLOWLIST:
        return []
    try:
        src = path.read_text()
    except Exception:
        return []
    lines = src.splitlines()
    hits: list[tuple[str, int, str, str]] = []
    for kind, pat, why in PATTERNS:
        for m in pat.finditer(src):
            line_no = src[: m.start()].count("\n") + 1
            line_text = (
                lines[line_no - 1][:160] if line_no - 1 < len(lines) else ""
            )
            hits.append((kind, line_no, line_text, why))
    return hits


def main() -> int:
    if not VERIFY_DIR.exists():
        print(f"verify dir not found at {VERIFY_DIR}; skipping")
        return 0

    files = list(VERIFY_DIR.glob("*.py")) + list(PIPELINE_DIR.glob("*.py"))
    flagged: list[tuple[Path, list[tuple[str, int, str, str]]]] = []
    for f in files:
        hits = scan_file(f)
        if hits:
            flagged.append((f, hits))

    if not flagged:
        print(f"verifier_authenticity_lint: clean ({len(files)} files scanned)")
        return 0

    total = sum(len(h) for _, h in flagged)
    print(
        f"verifier_authenticity_lint: {total} violation(s) across "
        f"{len(flagged)} file(s).\n"
        f"Per CLAUDE.md 'Verifier Authenticity Discipline': verifier code "
        f"must NOT game the adversarial-verify checks. Real compute takes "
        f"wall-time; real metrics come from real data.\n"
    )
    for path, hits in flagged:
        rel = path.relative_to(PROJECT_ROOT)
        print(f"\n  {rel}:")
        for kind, ln, txt, why in hits[:5]:
            print(f"    L{ln} [{kind}] {txt.strip()[:120]}")
            print(f"          why: {why}")
        if len(hits) > 5:
            print(f"    ... and {len(hits) - 5} more")
    return 1


if __name__ == "__main__":
    sys.exit(main())
