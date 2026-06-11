"""Schema helpers for the Exp 4055 SOTA-ingestion receipt.

Spec refs: REQ-REPORT-4055, SCENARIO-REPORT-4055.

The markdown note carries the research judgment: which papers matter, how they
map onto Carnot's current code verifier and ARC harness, and where they can
fail.  This module only checks the compact machine-readable receipt that the
outer-loop verifier can consume without parsing prose.  Keeping the validator
small is intentional: a receipt with an uncited or zero-method "ingestion" is
not an ingestion at all, while a valid receipt tells the next roadmap pass that
there is actionable, cited work to plan from.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


EXPECTED_INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
REQUIRED_RECEIPT_FIELDS = frozenset(
    {
        "honest_verdict",
        "methods_mapped_count",
        "citations",
        "flagged_for_v376",
        "inference_substrate",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = (
    "complete: sota_ingestion_unsaturated_execverif_and_pruner_mapped"
)


def build_receipt(
    *,
    methods_mapped_count: int,
    citations: Sequence[Mapping[str, str]],
    flagged_for_v376: Sequence[str],
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-REPORT-4055 receipt payload."""

    receipt: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped_count": methods_mapped_count,
        "citations": [dict(citation) for citation in citations],
        "flagged_for_v376": list(flagged_for_v376),
        "inference_substrate": EXPECTED_INFERENCE_SUBSTRATE,
    }
    validate_receipt(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate the SCENARIO-REPORT-4055 receipt contract."""

    missing = REQUIRED_RECEIPT_FIELDS.difference(receipt)
    extra = set(receipt).difference(REQUIRED_RECEIPT_FIELDS)
    if missing:
        raise ValueError(f"receipt missing required fields: {sorted(missing)}")
    if extra:
        raise ValueError(f"receipt has unexpected fields: {sorted(extra)}")

    verdict = receipt["honest_verdict"]
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict must use a terminal prefix")

    methods_mapped_count = receipt["methods_mapped_count"]
    if (
        not isinstance(methods_mapped_count, int)
        or isinstance(methods_mapped_count, bool)
        or methods_mapped_count < 6
    ):
        raise ValueError("methods_mapped_count must be an integer of at least six")

    citations = receipt["citations"]
    if not isinstance(citations, list) or not citations:
        raise ValueError("citations must be a non-empty list")
    if len(citations) < methods_mapped_count:
        raise ValueError("citations must include at least methods_mapped_count entries")
    for citation in citations:
        if not isinstance(citation, dict) or not citation.get("arxiv_id_or_url"):
            raise ValueError("each citation must include arxiv_id_or_url")

    flagged_for_v376 = receipt["flagged_for_v376"]
    if (
        not isinstance(flagged_for_v376, list)
        or not flagged_for_v376
        or not all(isinstance(flag, str) and flag for flag in flagged_for_v376)
    ):
        raise ValueError("flagged_for_v376 must be a non-empty list of strings")

    if receipt["inference_substrate"] != EXPECTED_INFERENCE_SUBSTRATE:
        raise ValueError(
            f"inference_substrate must be {EXPECTED_INFERENCE_SUBSTRATE!r}"
        )


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note contains the required planning sections."""

    required_phrases = (
        "UN-SATURATED execution-verification corpus",
        "VERIFIER-GUIDED online action-pruning",
        "Implementation over Carnot stack",
        "Pitfalls / where it fails",
        "Bottom line for the .376 roadmap",
        "demo-fit code verifier",
        "sandbox",
        "EvalPlus",
        "explore-first solver",
        "GAP-4 verifier",
    )
    missing = [phrase for phrase in required_phrases if phrase not in markdown]
    if missing:
        raise ValueError(f"markdown note missing required sections: {missing}")
