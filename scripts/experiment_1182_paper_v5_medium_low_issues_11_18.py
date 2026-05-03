#!/usr/bin/env python3
"""Experiment 1182 — Paper v5 medium/low integrity fixes (ISSUE-11..18).

Spec traces: REQ-PUBLISH-008, REQ-PUBLISH-009, SCENARIO-PUBLISH-008.
"""

from __future__ import annotations

import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import paper_claim_audit

_PAPER = _REPO_ROOT / "docs" / "arxiv-paper" / "main.tex"
_BIB = _REPO_ROOT / "docs" / "arxiv-paper" / "carnot.bib"
_RESULTS = _REPO_ROOT / "results"
_DELIVERABLE = _RESULTS / "experiment_1182_paper_v5_medium_low_issues_11_18.json"


def _load_paper() -> str:
    """Read the arXiv paper source."""
    return _PAPER.read_text(encoding="utf-8")


def _load_bib() -> str:
    """Read the arXiv bibliography source."""
    return _BIB.read_text(encoding="utf-8")


def check_issue_11_thinkprm(tex: str) -> bool:
    """Return True when ThinkPRM AUROC=0.9885 names the exp1111 lineage."""
    return bool(
        "ThinkPRM" in tex
        and "0.9885" in tex
        and re.search(r"exp1111\s+v1", tex)
        and "0.9946" in tex
        and "experiment\\_1111\\_thinkprm\\_retrain.json" in tex
    )


def check_issue_12_holdout(tex: str) -> bool:
    """Return True when the n=50 holdout and exp1121 contradiction are disclosed."""
    compact = " ".join(tex.split())
    return bool(
        "n=50 holdout" in compact
        and "exp1121" in compact
        and "0.3333" in compact
        and "production corpus" in compact
        and "0.9545" in compact
        and "FoVer holdout" in compact
    )


def check_issue_13_nrgpt(tex: str) -> tuple[bool, str]:
    """Return (resolved, status) for the NRGPT non-monotonicity disclosure."""
    if not re.search(r"NRGPT|NR-GPT|nrgpt|n_iters", tex):
        return True, "not_cited"
    disclosed = bool(
        "AUROC_n1=0.9209" in tex
        and "AUROC_n3=0.9158" in tex
        and "n_iters_monotone=False" in tex
        and "energy iteration NOT monotone" in tex
    )
    return disclosed, "disclosed" if disclosed else "missing"


def _windows_around(tex: str, token: str, radius: int = 220) -> list[str]:
    """Return local windows around every occurrence of ``token``."""
    return [tex[max(0, m.start() - radius) : m.end() + radius] for m in re.finditer(token, tex)]


def _has_sample_size(window: str, *sizes: str) -> bool:
    """Return True if a window names at least one accepted sample-size spelling."""
    compact = window.replace(" ", "")
    return any(size.replace(" ", "") in compact for size in sizes)


def check_issue_14_soskan_auroc(tex: str) -> bool:
    """Return True when SOS-KAN/SOSKAN AUROC values identify corpus and sample size."""
    for window in _windows_around(tex, "0.9545"):
        if "SOS" in window or "energy function" in window or "FoVer holdout" in window:
            if "FoVer" not in window:
                return False
            if not _has_sample_size(window, "n=6{,}548", "6{,}548-pair", "n=50"):
                return False

    for window in _windows_around(tex, "0.9902"):
        if "SOS" in window or "SOSKAN" in window:
            if "production" not in window or not _has_sample_size(window, "n=500"):
                return False

    for window in _windows_around(tex, "0.9774"):
        if "SOS" in window or "retrain" in window:
            if "7{,}329" not in window or "exp1120" not in window:
                return False

    return True


def check_issue_15_fig2_caveat(tex: str) -> bool:
    """Return True when Figure 2's caption includes the binormal-fit caveat."""
    return "Binormal fit from published AUROC; not a re-evaluation on" in tex


def check_issue_16_bibliography(tex: str, bib: str) -> tuple[int, bool]:
    """Return (removed_stub_count, ok) for the suspect bibliography entries."""
    themesis_removed = "themesis2026seediq" not in bib and "themesis2026seediq" not in tex
    removed_count = 1 if themesis_removed else 0

    real_entries_have_authors = all(
        token in bib
        for token in [
            "@article{rewardunderattack2026",
            "Tiwari, Rishabh",
            "Analyzing the Robustness and Hackability",
            "@article{llmsgamingverifiers2026",
            "Helff, Lukas",
            "RLVR can Lead to Reward Hacking",
            "@article{hive2026",
            "Zhao, Guoshenghui",
        ]
    )
    return removed_count, bool(themesis_removed and real_entries_have_authors)


def check_issue_17_k15_caption(tex: str) -> bool:
    """Return True when Table 1 clarifies the k=15 row's theoretical status."""
    compact = " ".join(tex.replace("$", "").split())
    return bool(
        "k=15 row is the theoretical maximum from the AND-composition bound" in compact
        and "Theorem~3.2" in compact
        and "exp1108" in compact
        and "k=15 is not an experimentally achieved result" in compact
    )


def check_issue_18_hardware_scope(tex: str) -> bool:
    """Return True when hardware portability is scoped to measured KV260 evidence."""
    compact = " ".join(tex.split())
    return bool(
        "no substrate other than the KV260 FPGA has been empirically verified" in compact
        and "portability to Extropic Z1 and other platforms remains planned future work" in compact
    )


def run() -> dict:
    """Run all exp1182 checks and return the artifact payload."""
    tex = _load_paper()
    bib = _load_bib()

    issue_13_resolved, issue_13_status = check_issue_13_nrgpt(tex)
    issue_16_removed, issue_16_ok = check_issue_16_bibliography(tex, bib)
    issue_checks = {
        "issue_11_thinkprm_citation_fixed": check_issue_11_thinkprm(tex),
        "issue_12_holdout_n_stated": check_issue_12_holdout(tex),
        "issue_13_nrgpt_disclosure_added": issue_13_resolved,
        "issue_14_soskan_auroc_reconciled": check_issue_14_soskan_auroc(tex),
        "issue_15_fig2_caveat_added": check_issue_15_fig2_caveat(tex),
        "issue_16_bibliography_ok": issue_16_ok,
        "issue_17_k15_caption_tightened": check_issue_17_k15_caption(tex),
        "issue_18_hardware_scope_added": check_issue_18_hardware_scope(tex),
    }
    medium_low_issues_fixed = sum(bool(value) for value in issue_checks.values())

    claim_report = paper_claim_audit.audit_paper_claims(_PAPER, _RESULTS)
    all_resolved = medium_low_issues_fixed == 8 and claim_report["passes"]
    honest_verdict = (
        "all_8_medium_low_resolved"
        if all_resolved
        else "partial_fix"
        if medium_low_issues_fixed > 0
        else "blocked"
    )

    return {
        "experiment": 1182,
        "title": "Paper v5 medium/low integrity fixes (ISSUE-11 through ISSUE-18)",
        "run_date": datetime.now(UTC).isoformat(),
        "schema": "experiment_result_v1",
        "issue_11_thinkprm_citation_fixed": issue_checks["issue_11_thinkprm_citation_fixed"],
        "issue_12_holdout_n_stated": issue_checks["issue_12_holdout_n_stated"],
        "issue_13_nrgpt_disclosure_added": issue_checks["issue_13_nrgpt_disclosure_added"],
        "issue_13_nrgpt_status": issue_13_status,
        "issue_14_soskan_auroc_reconciled": issue_checks["issue_14_soskan_auroc_reconciled"],
        "issue_15_fig2_caveat_added": issue_checks["issue_15_fig2_caveat_added"],
        "issue_16_bib_stubs_removed": issue_16_removed,
        "issue_17_k15_caption_tightened": issue_checks["issue_17_k15_caption_tightened"],
        "issue_18_hardware_scope_added": issue_checks["issue_18_hardware_scope_added"],
        "paper_claim_audit_script_active": claim_report["passes"],
        "paper_claim_audit_n_claims_total": claim_report["n_claims_total"],
        "paper_claim_audit_n_verified": claim_report["n_claims_verified"],
        "paper_claim_audit_n_mismatches": claim_report["n_mismatches"],
        "paper_claim_audit_n_claims_with_artifact_citation": claim_report[
            "n_claims_with_artifact_citation"
        ],
        "paper_claim_audit_citation_ratio": claim_report["citation_ratio"],
        "medium_low_issues_fixed": medium_low_issues_fixed,
        "honest_verdict": honest_verdict,
        "status": "success" if all_resolved else "partial",
        "duration_s": 0.0,
    }


def main() -> None:
    """Write the exp1182 deliverable JSON and exit nonzero on partial fixes."""
    import time

    t0 = time.monotonic()
    payload = run()
    payload["duration_s"] = round(time.monotonic() - t0, 3)

    _DELIVERABLE.parent.mkdir(parents=True, exist_ok=True)
    _DELIVERABLE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if payload["honest_verdict"] != "all_8_medium_low_resolved":
        print(
            f"\nWARNING: not all fixes verified. medium_low_issues_fixed="
            f"{payload['medium_low_issues_fixed']}/8",
            file=sys.stderr,
        )
        raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
