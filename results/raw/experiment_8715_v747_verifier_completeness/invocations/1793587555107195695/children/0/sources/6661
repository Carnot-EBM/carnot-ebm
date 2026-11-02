#!/usr/bin/env python3
"""Experiment 1181 — Paper v5 high-severity integrity fixes (ISSUE-6 through ISSUE-10).

**What this does:**
    Verifies that all five high-severity integrity issues identified in the paper-v5
    audit have been applied to docs/arxiv-paper/main.tex, then writes the
    canonical result artifact.

**Why these five issues matter:**
    High-severity issues do not individually block arXiv submission but all five
    must be resolved before the operator approves hold-lift.  The audit is defined in
    openspec/change-proposals/paper-v5-integrity-remediation.md Phase 2 section.

    ISSUE-6: GRPO delta claims lacked sample sizes and binomial CIs; n=25 CIs span
             near-zero, so presenting +8.51 pp without error bars overstates precision.
    ISSUE-7: HumanEval baseline 0.0% was a harness extraction failure, not a model
             accuracy score; presenting it as a headline result misleads readers.
    ISSUE-8: alpha_t=0.38 omitted the 24/100 false-rejection rate, hiding that the
             Zenil positivity condition is satisfied but calibration to ground truth
             still requires work.
    ISSUE-9: Phase-4 pilot used a random-legal-greedy baseline (trivially weak);
             74.7% action reduction against it overstates the result.
    ISSUE-10: Seed IQ score cited from a published announcement without independent
              verification (exp1166: seed_iq_score_confirmed=false); this must be
              disclosed as a documented external reference, not a reproduced result.

Spec: REQ-PUBLISH-007, SCENARIO-PUBLISH-007
"""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone, UTC
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PAPER = _REPO_ROOT / "docs" / "arxiv-paper" / "main.tex"
_DELIVERABLE = _REPO_ROOT / "results" / "experiment_1181_paper_v5_high_issues_6_10.json"


# ---------------------------------------------------------------------------
# Verification helpers
# ---------------------------------------------------------------------------


def _load_paper() -> str:
    """Read main.tex and return its full content as a string."""
    return _PAPER.read_text(encoding="utf-8")


def find_unannotated_grpo_delta_claims(tex: str) -> list[str]:
    """Return GRPO delta claim blocks missing inline sample size and CI annotations.

    Spec traces: REQ-PUBLISH-007, SCENARIO-PUBLISH-007.
    The paper has both abstract and body prose; splitting on blank lines gives
    enough local context to require each GRPO delta paragraph to include one
    ``(n=..., 95% CI: [...])`` annotation per delta token.
    """
    missing: list[str] = []
    delta_tokens = ("+4", "+8.51", "+10.0", "+2.86")

    for block in re.split(r"\n\s*\n", tex):
        normalized = block.replace("$", "").replace(r"\%", "%")
        if "GRPO" not in normalized or "pp" not in normalized:
            continue

        delta_count = sum(1 for token in delta_tokens if token in normalized)
        if delta_count == 0:
            continue

        ci_count = len(re.findall(r"n=\d+,\s*95% CI:\s*\[[^\]]+\]", normalized))
        if ci_count < delta_count:
            missing.append(" ".join(block.split())[:240])

    return missing


def check_issue_6_grpo_cis(tex: str) -> tuple[bool, bool]:
    """Return (cis_added, caveat_added) for ISSUE-6.

    CIs are added when the text contains Clopper-Pearson-style bracketed
    percentage ranges adjacent to the GRPO delta claims.
    The small-sample caveat is present when the footnote warning text appears.
    """
    # Look for CI annotation pattern "[X%, Y%]" near every GRPO delta text block.
    cis_added = not find_unannotated_grpo_delta_claims(tex)
    # Look for the small-sample footnote
    caveat_added = bool(
        re.search(r"small evaluation sets.*n=25", tex, re.DOTALL)
        or re.search(r"preliminary indicators.*not definitive accuracy", tex, re.DOTALL)
    )
    return cis_added, caveat_added


def check_issue_7_humaneval(tex: str) -> bool:
    """Return True when HumanEval 0.0% is framed as a harness extraction failure."""
    return bool(
        re.search(r"harness extraction failure", tex, re.IGNORECASE)
        and re.search(r"extraction.fix.*\+36.*pp|\+36.*pp.*extraction.fix", tex, re.DOTALL)
        and r"\label{app:harness-anomalies}" in tex
    )


def check_issue_8_alpha_t(tex: str) -> bool:
    """Return True when the 24/100 false-rejection rate is disclosed near alpha_t=0.38.

    The LaTeX text spans multiple lines, so we search for the key phrases
    independently rather than requiring them on one line.
    """
    return bool(
        re.search(r"false rejection\b", tex, re.IGNORECASE)
        and re.search(r"rate 24", tex, re.IGNORECASE)
        and re.search(r"ground.truth.correct", tex, re.DOTALL)
        and re.search(r"24.*rejected", tex, re.DOTALL)
    )


def check_issue_9_phase4_baseline(tex: str) -> bool:
    """Return True when the Phase-4 pilot caveat and exp1189 forward reference are present."""
    return bool(
        re.search(
            r"random.legal.greedy.*intentionally weak|intentionally weak.*random.legal.greedy",
            tex,
            re.DOTALL,
        )
        and re.search(r"exp1189", tex)
    )


def check_issue_10_seed_iq(tex: str) -> bool:
    """Return True when the Seed IQ footnote disclosing non-verification is present.

    In LaTeX, underscores in text are escaped as \\_, so the file contains
    ``seed\\_iq\\_score\\_confirmed=false``.  We match the key token fragments
    independently rather than the full escaped form.
    """
    # Match the exp1166 seed_iq_score_confirmed=false note (LaTeX-escaped or plain)
    has_confirmed_false = bool(
        re.search(r"seed.*iq.*score.*confirmed\s*=\s*false", tex, re.IGNORECASE | re.DOTALL)
        or re.search(r"seed_iq_score_confirmed=false", tex, re.IGNORECASE)
    )
    has_not_refetched = bool(
        re.search(r"not\s+independently\s+re.fetched", tex, re.IGNORECASE | re.DOTALL)
    )
    return has_confirmed_false and has_not_refetched


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run() -> dict:
    """Execute all five checks and return the result payload."""
    tex = _load_paper()

    ci_added, caveat_added = check_issue_6_grpo_cis(tex)
    humaneval_reframed = check_issue_7_humaneval(tex)
    alpha_t_disclosed = check_issue_8_alpha_t(tex)
    phase4_caveat = check_issue_9_phase4_baseline(tex)
    seed_iq_footnote = check_issue_10_seed_iq(tex)

    fixed_count = sum(
        [
            ci_added,
            caveat_added,
            humaneval_reframed,
            alpha_t_disclosed,
            phase4_caveat,
            seed_iq_footnote,
        ]
    )
    # 6 booleans map to 5 issues (ISSUE-6 has two sub-checks; both must pass)
    # high_severity_fixed counts distinct issues resolved:
    issue6_resolved = ci_added and caveat_added
    high_severity_fixed = sum(
        [
            issue6_resolved,
            humaneval_reframed,
            alpha_t_disclosed,
            phase4_caveat,
            seed_iq_footnote,
        ]
    )

    all_resolved = high_severity_fixed == 5
    honest_verdict = (
        "all_5_high_resolved"
        if all_resolved
        else "partial_fix"
        if high_severity_fixed > 0
        else "blocked"
    )

    return {
        "experiment": 1181,
        "title": "Paper v5 high-severity integrity fixes (ISSUE-6 through ISSUE-10)",
        "run_date": datetime.now(UTC).isoformat(),
        "schema": "experiment_result_v1",
        "issue_6_grpo_cis_added": ci_added,
        "issue_6_small_sample_caveat_added": caveat_added,
        "issue_7_humaneval_reframed": humaneval_reframed,
        "issue_8_alpha_t_rejection_rate_added": alpha_t_disclosed,
        "issue_9_phase4_baseline_caveat_added": phase4_caveat,
        "issue_10_seed_iq_footnote_added": seed_iq_footnote,
        "high_severity_fixed": high_severity_fixed,
        "4_test_passes_high": all_resolved,
        "honest_verdict": honest_verdict,
        "status": "success" if all_resolved else "partial",
        "duration_s": 0.0,
    }


def main() -> None:
    """Write the deliverable JSON and exit."""
    import time

    t0 = time.monotonic()
    payload = run()
    payload["duration_s"] = round(time.monotonic() - t0, 3)

    _DELIVERABLE.parent.mkdir(parents=True, exist_ok=True)
    _DELIVERABLE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(payload, indent=2))
    if payload["honest_verdict"] != "all_5_high_resolved":
        print(
            f"\nWARNING: not all fixes verified. high_severity_fixed={payload['high_severity_fixed']}/5",
            file=sys.stderr,
        )
        sys.exit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
