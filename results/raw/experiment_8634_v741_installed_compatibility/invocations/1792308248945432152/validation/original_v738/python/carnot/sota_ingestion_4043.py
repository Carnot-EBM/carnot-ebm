"""Schema helpers for the Exp 4043 SOTA-ingestion receipt.

Spec refs: REQ-PHASE4-039, SCENARIO-PHASE4-039.

The markdown note carries the research judgment.  This module checks the small
machine-readable contract that downstream adversarial verification expects:
terminal verdict, nonzero mapped methods, cited sources, roadmap flags, and the
document-aggregation inference substrate.
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
        "flagged_for_v375",
        "inference_substrate",
    }
)
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "blocked_")
DEFAULT_HONEST_VERDICT = (
    "complete: sota_ingestion_offarc_power_and_closed_loop_mapped"
)


def build_receipt(
    *,
    methods_mapped_count: int,
    citations: Sequence[Mapping[str, str]],
    flagged_for_v375: Sequence[str],
    honest_verdict: str = DEFAULT_HONEST_VERDICT,
) -> dict[str, object]:
    """Build and validate the REQ-PHASE4-039 receipt payload."""

    receipt: dict[str, object] = {
        "honest_verdict": honest_verdict,
        "methods_mapped_count": methods_mapped_count,
        "citations": [dict(citation) for citation in citations],
        "flagged_for_v375": list(flagged_for_v375),
        "inference_substrate": EXPECTED_INFERENCE_SUBSTRATE,
    }
    validate_receipt(receipt)
    return receipt


def validate_receipt(receipt: Mapping[str, Any]) -> None:
    """Validate the SCENARIO-PHASE4-039 receipt contract."""

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
    if not isinstance(methods_mapped_count, int) or methods_mapped_count < 3:
        raise ValueError("methods_mapped_count must be an integer of at least three")

    citations = receipt["citations"]
    if not isinstance(citations, list) or not citations:
        raise ValueError("citations must be a non-empty list")
    for citation in citations:
        if not isinstance(citation, dict) or not citation.get("arxiv_id_or_url"):
            raise ValueError("each citation must include arxiv_id_or_url")

    flagged_for_v375 = receipt["flagged_for_v375"]
    if (
        not isinstance(flagged_for_v375, list)
        or not flagged_for_v375
        or not all(isinstance(flag, str) and flag for flag in flagged_for_v375)
    ):
        raise ValueError("flagged_for_v375 must be a non-empty list of strings")

    if receipt["inference_substrate"] != EXPECTED_INFERENCE_SUBSTRATE:
        raise ValueError(
            f"inference_substrate must be {EXPECTED_INFERENCE_SUBSTRATE!r}"
        )


def validate_markdown_note(markdown: str) -> None:
    """Check that the paired note contains the required planning sections."""

    required_phrases = (
        "OFF-ARC power + stronger discriminator",
        "CLOSED-LOOP planning over a verified world model",
        "Implementation over Carnot stack",
        "Pitfalls / where it fails",
        "Bottom line for the .375 roadmap",
    )
    missing = [phrase for phrase in required_phrases if phrase not in markdown]
    if missing:
        raise ValueError(f"markdown note missing required sections: {missing}")
