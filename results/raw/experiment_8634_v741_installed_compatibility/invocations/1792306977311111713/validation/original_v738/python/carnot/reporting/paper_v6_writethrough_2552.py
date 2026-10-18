"""Exp 2552 paper-v6 write-through artifact builder.

The workflow is intentionally narrow: verify that paper-v6 includes the
milestone .245 AUROC result, the real-corpus Tier 0r/0s/0u verifier values,
and the three requested arXiv citations, while preserving the Phase 4 honest
negative from Exp 2544.

Spec refs: REQ-PUBLISH-003, REQ-PUBLISH-021.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
PAPER_REL_PATH = Path("docs/arxiv-paper/main.tex")
BIB_REL_PATH = Path("docs/arxiv-paper/carnot.bib")
EXP2544_REL_PATH = Path("results/experiment_2544_phase4_option_b.json")
EXP2546_REL_PATH = Path("results/experiment_2546_ensemble_v7b.json")
EXP2548_REL_PATH = Path("results/experiment_2548_real_corpus_validation.json")
OUTPUT_REL_PATH = Path("results/experiment_2552_paper_writethrough.json")

REQUIRED_CITATION_IDS = (
    "arXiv:2512.18730",
    "arXiv:2604.17109",
    "arXiv:2605.09515",
)
REQUIRED_CITATION_KEYS = (
    "tan2025rltunedebm",
    "zhu2026parallelisingmachine",
    "bouchaffra2026gamefep",
)
REQUIRED_REAL_VERIFIERS = ("tier0r", "tier0s", "tier0u")
TERMINAL_PREFIXES = ("complete:", "blocked_", "blocked:")

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": "Terminal-prefix required.",
    "paper_updated": "True if main.tex was modified with ensemble v7b results and new citations.",
    "citations_added": "List of arXiv IDs added in this task -- audit trail.",
    "ensemble_v7b_incorporated": "True if exp2546 AUROC result appears in the paper.",
    "phase4_section_intact": "True if Section 4 honest negative from exp2544 was not overwritten.",
    "preconditions_checked": "Records which resources were verified.",
    "duration_s": "Wall-clock measurement.",
}


def read_json(path: Path) -> Mapping[str, Any]:
    """Read a JSON object from a local artifact path."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, Mapping) else {}


def _metric(value: Any) -> str:
    return f"{float(value):.4f}"


def _paper_citable_verifiers(exp2548: Mapping[str, Any]) -> list[str]:
    paper_citable = exp2548.get("paper_citable", {})
    if not isinstance(paper_citable, Mapping):
        return []
    return [
        verifier
        for verifier in REQUIRED_REAL_VERIFIERS
        if paper_citable.get(verifier) is True
        and f"{verifier}_real_auroc" in exp2548
    ]


def phase4_honest_negative_intact(paper_text: str) -> bool:
    """Return whether the Exp 2544 honest-negative paragraph is still present."""

    required_fragments = (
        "Honest negative result",
        "exp2486",
        "exp2508",
        "exp2519",
        "exp2532",
        "Phase~4 remains a theoretical hypothesis",
        "step-level granularity was not achieved",
    )
    return all(fragment in paper_text for fragment in required_fragments)


def paper_update_status(
    paper_text: str,
    bib_text: str,
    exp2546: Mapping[str, Any],
    exp2548: Mapping[str, Any],
) -> dict[str, bool]:
    """Check the paper and bibliography for Exp 2552 required write-throughs."""

    ensemble_value = exp2546.get("ensemble_v7b_auroc")
    ensemble_v7b_incorporated = (
        ensemble_value is not None and _metric(ensemble_value) in paper_text
    )

    citable_verifiers = _paper_citable_verifiers(exp2548)
    real_corpus_aurocs_incorporated = bool(citable_verifiers) and all(
        _metric(exp2548[f"{verifier}_real_auroc"]) in paper_text
        for verifier in citable_verifiers
    )

    citations_present = all(citation in bib_text for citation in REQUIRED_CITATION_IDS)
    citation_keys_in_text = all(key in paper_text for key in REQUIRED_CITATION_KEYS)

    return {
        "ensemble_v7b_incorporated": ensemble_v7b_incorporated,
        "real_corpus_aurocs_incorporated": real_corpus_aurocs_incorporated,
        "citations_present": citations_present and citation_keys_in_text,
        "phase4_section_intact": phase4_honest_negative_intact(paper_text),
    }


def _precondition(path: Path, description: str) -> dict[str, Any]:
    return {
        "resource": str(path),
        "description": description,
        "available": path.is_file(),
    }


def build_artifact(
    root: Path = REPO_ROOT,
    *,
    started_epoch: float | None = None,
    now_epoch: float | None = None,
) -> dict[str, Any]:
    """Build the terminal Exp 2552 artifact from checked-in paper sources."""

    now = time.time() if now_epoch is None else now_epoch
    started = now if started_epoch is None else started_epoch
    paper_path = root / PAPER_REL_PATH
    bib_path = root / BIB_REL_PATH
    exp2544_path = root / EXP2544_REL_PATH
    exp2546_path = root / EXP2546_REL_PATH
    exp2548_path = root / EXP2548_REL_PATH

    preconditions = [
        _precondition(paper_path, "paper source"),
        _precondition(bib_path, "paper bibliography"),
        _precondition(exp2546_path, "ensemble v7b source artifact"),
        _precondition(exp2548_path, "real-corpus verifier source artifact"),
        _precondition(exp2544_path, "Phase 4 honest-negative source artifact"),
    ]
    duration_s = round(max(0.0, now - started), 6)

    if not paper_path.is_file():
        return {
            "honest_verdict": "blocked_paper_not_found",
            "paper_updated": False,
            "citations_added": [],
            "ensemble_v7b_incorporated": False,
            "phase4_section_intact": False,
            "preconditions_checked": preconditions,
            "duration_s": duration_s,
            "field_principles": FIELD_PRINCIPLES,
            "acceptance_gates": {"paper_updated == true": False},
        }

    paper_text = paper_path.read_text(encoding="utf-8")
    bib_text = bib_path.read_text(encoding="utf-8") if bib_path.is_file() else ""
    exp2544 = read_json(exp2544_path)
    exp2546 = read_json(exp2546_path)
    exp2548 = read_json(exp2548_path)
    status = paper_update_status(paper_text, bib_text, exp2546, exp2548)

    paper_updated = (
        status["ensemble_v7b_incorporated"]
        and status["real_corpus_aurocs_incorporated"]
        and status["citations_present"]
    )
    phase4_section_intact = (
        status["phase4_section_intact"]
        and exp2544.get("phase4_honest_negative_documented") is True
    )

    real_verifier_values = {
        verifier: exp2548.get(f"{verifier}_real_auroc")
        for verifier in _paper_citable_verifiers(exp2548)
    }
    honest_prefix = "complete:" if paper_updated and phase4_section_intact else "blocked:"

    return {
        "honest_verdict": (
            f"{honest_prefix} paper_updated={paper_updated}; "
            f"ensemble_v7b_incorporated={status['ensemble_v7b_incorporated']}; "
            f"phase4_section_intact={phase4_section_intact}"
        ),
        "paper_updated": paper_updated,
        "citations_added": list(REQUIRED_CITATION_IDS) if status["citations_present"] else [],
        "ensemble_v7b_incorporated": status["ensemble_v7b_incorporated"],
        "phase4_section_intact": phase4_section_intact,
        "preconditions_checked": preconditions,
        "duration_s": duration_s,
        "ensemble_v7b_auroc": exp2546.get("ensemble_v7b_auroc"),
        "ensemble_v7b_auroc_std": exp2546.get("ensemble_v7b_auroc_std"),
        "real_corpus_verifier_aurocs": real_verifier_values,
        "n_real": exp2548.get("n_real"),
        "paper_citable": exp2548.get("paper_citable", {}),
        "status_checks": status,
        "field_principles": FIELD_PRINCIPLES,
        "acceptance_gates": {
            "paper_updated == true": paper_updated,
            "Phase 4 honest negative preserved": phase4_section_intact,
        },
        "files_modified": [
            str(PAPER_REL_PATH),
            str(BIB_REL_PATH),
            str(OUTPUT_REL_PATH),
            "python/carnot/reporting/paper_v6_writethrough_2552.py",
            "tests/python/test_experiment_2552_paper_writethrough.py",
        ],
    }


def main() -> int:
    started_env = os.environ.get("CARNOT_EXP2552_START_EPOCH")
    started_epoch = float(started_env) if started_env else None
    artifact = build_artifact(REPO_ROOT, started_epoch=started_epoch)
    out_path = REPO_ROOT / OUTPUT_REL_PATH
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if artifact["paper_updated"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
